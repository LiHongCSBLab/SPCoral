import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

class GATConv(nn.Module):
    """Adapted from STAGATE <https://doi.org/10.1038/s41467-022-29439-6>"""
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0.0, add_self_loops=True, bias=True):
        super(GATConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        # Linear transformations for source and destination nodes
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        # Attention parameters
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.attentions = None
        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_normal_(self.lin.weight, gain=1.414)
        nn.init.xavier_normal_(self.att_src, gain=1.414)
        nn.init.xavier_normal_(self.att_dst, gain=1.414)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, g, h, attention=True, tied_attention=None, return_attention_weights=False):
        """
        Args:
            g (dgl.DGLGraph): The graph.
            h (torch.Tensor): Node features of shape (num_nodes, in_channels).
            attention (bool): If False, skip attention and return transformed features.
            tied_attention (tuple): Precomputed attention coefficients (optional).
            return_attention_weights (bool): If True, return edge attention weights.

        Returns:
            torch.Tensor: Output node features.
            tuple: (edge_ids, attention_weights) if return_attention_weights is True.
        """
        H, C = self.heads, self.out_channels

        # Add self-loops if required
        if self.add_self_loops:
            g = dgl.add_self_loop(g)

        # Transform node features
        h = self.lin(h).view(-1, H, C)  # (num_nodes, heads, out_channels)

        if not attention:
            return h.mean(dim=1) if not self.concat else h.view(-1, H * C)

        # Compute attention coefficients
        if tied_attention is None:
            alpha_src = (h * self.att_src).sum(dim=-1)  # (num_nodes, heads)
            alpha_dst = (h * self.att_dst).sum(dim=-1)  # (num_nodes, heads)
            self.attentions = (alpha_src, alpha_dst)
        else:
            alpha_src, alpha_dst = tied_attention

        # Store features and attention coefficients in graph
        g.ndata['h'] = h  # (num_nodes, heads, out_channels)
        g.ndata['alpha_src'] = alpha_src  # (num_nodes, heads)
        g.ndata['alpha_dst'] = alpha_dst  # (num_nodes, heads)

        # Message function: Compute edge-wise attention scores
        def message_func(edges):
            alpha = edges.src['alpha_src'] + edges.dst['alpha_dst']  # (num_edges, heads)
            # alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = torch.sigmoid(alpha)
            return {'alpha': alpha, 'h': edges.src['h']}

        # Reduce function: Normalize attention scores and aggregate
        def reduce_func(nodes):
            alpha = nodes.mailbox['alpha']  # (num_nodes, num_neighbors, heads)
            h = nodes.mailbox['h']  # (num_nodes, num_neighbors, heads, out_channels)
            alpha = F.softmax(alpha, dim=1)  # Normalize over neighbors
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            out = (h * alpha.unsqueeze(-1)).sum(dim=1)  # (num_nodes, heads, out_channels)
            return {'h': out}

        # Perform message passing
        g.update_all(message_func, reduce_func)
        out = g.ndata.pop('h')  # (num_nodes, heads, out_channels)
        self._alpha = g.ndata.pop('alpha_src')  # Save for potential return

        # Concatenate or average heads
        if self.concat:
            out = out.view(-1, H * C)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if return_attention_weights:
            # Compute edge attention weights for return
            with g.local_scope():
                g.ndata['alpha_src'] = alpha_src
                g.ndata['alpha_dst'] = alpha_dst
                g.apply_edges(fn.u_add_v('alpha_src', 'alpha_dst', 'alpha'))
                alpha = torch.sigmoid(g.edata['alpha'])
                edge_ids = g.edges()
                return out, (edge_ids, alpha)

        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, heads={self.heads})'