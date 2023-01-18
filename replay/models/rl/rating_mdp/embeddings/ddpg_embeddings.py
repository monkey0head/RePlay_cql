from torch import nn
import torch
class StateReprModule(nn.Module):
    """
    Compute state for RL environment. Based on `DRR paper
    <https://arxiv.org/pdf/1810.12027.pdf>`_

    Computes State is a concatenation of user embedding,
    weighted average pooling of `memory_size` latest relevant items
    and their pairwise product.
    """

    def __init__(
        self,
        user_num,
        item_num,
        embedding_dim,
        memory_size,
    ):
        super().__init__()
        self.user_embeddings = nn.Embedding(user_num, embedding_dim)
        self.item_embeddings = nn.Embedding(
            item_num + 1, embedding_dim, padding_idx=int(item_num)
        )
        self.drr_ave = torch.nn.Conv1d(
            in_channels=memory_size, out_channels=1, kernel_size=1
        )

        self.initialize()

    def initialize(self):
        """weight init"""
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        self.item_embeddings.weight.data[-1].zero_()

        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.uniform_(self.drr_ave.weight)

        self.drr_ave.bias.data.zero_()

    def forward(self, user, memory):
        """
        :param user: user batch
        :param memory: memory batch
        :return: vector of dimension 3 * embedding_dim
        """
        user_embedding = self.user_embeddings(user.long())

        item_embeddings = self.item_embeddings(memory.long())
        drr_ave = self.drr_ave(item_embeddings).squeeze(1)

        return torch.cat(
            (user_embedding, user_embedding * drr_ave, drr_ave), 1
        )


class ActorDRR(nn.Module):
    """
    DDPG Actor model (based on `DRR
    <https://arxiv.org/pdf/1802.05814.pdf>`_).
    """

    def __init__(
        self, user_num, item_num, embedding_dim, hidden_dim, memory_size
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self.state_repr = StateReprModule(
            user_num, item_num, embedding_dim, memory_size
        )

        self.initialize()

       # self.environment = Env(item_num, user_num, memory_size)
      #  self.test_environment = Env(item_num, user_num, memory_size)

    def initialize(self):
        """weight init"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)

    def forward(self, user, memory):
        """
        :param user: user batch
        :param memory: memory batch
        :return: output, vector of the size `embedding_dim`
        """
        state = self.state_repr(user, memory)
        return self.layers(state)

    def get_action(
        self,
        action_emb,
        items,
        return_scores=False,
    ):
        """
        :param action_emb: output of the .forward()
        :param items: items batch
        :param return_scores: whether to return scores of items
        :return: output, prediction (and scores if return_scores)
        """
        scores = torch.bmm(
            self.state_repr.item_embeddings(items).unsqueeze(0),
            action_emb.T.unsqueeze(0),
        ).squeeze(0)
        if return_scores:
            return scores, torch.gather(items, 0, scores.argmax(0))
        else:
            return torch.gather(items, 0, scores.argmax(0))
        
        
def load_embeddings(path_to_model = "model_final.pt", model_params=[943, 1682, 8, 16, 5]):
    the_model = ActorDRR(*model_params)
    the_model.load_state_dict(torch.load(path_to_model))
    user_emb = the_model.state_repr.user_embeddings.weight.data
    item_emb = the_model.state_repr.item_embeddings.weight.data
    return user_emb, item_emb