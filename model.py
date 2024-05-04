from typing import Tuple
import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_k: int | None = None,
        d_v: int | None = None,
    ) -> None:
        """
        Multi Head Attention block. It can be used as Self Attention, Masked Self Attention
        and Multi Head Attention that gets Q from Decoder and K & V from Encoder

        Args:
            num_heads (int): number of attention heads. Note that d_k must be divisible by num_heads
            d_model (int): total dim size for input embeddings and resulting tensor
            d_k (int): total dim size for Keys and Queries. Defaults to d_model
            d_v (int): total dim size for Values. Defaults to d_model
        """
        super().__init__()

        assert d_k % num_heads == 0, "d_k must be divisible by num_heads"

        if d_k is None:
            d_k = d_model
        if d_v is None:
            d_v = d_model

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_v = d_v
        self.d_k = d_k

        self.d_v_per_head = d_v // num_heads
        self.d_k_per_head = d_k // num_heads

        self.inv_sqrt_for_head = (self.d_k_per_head) ** (-0.5)

        self.fc_q = nn.Linear(
            in_features=self.d_model, out_features=self.d_k, bias=False
        )
        self.fc_k = nn.Linear(
            in_features=self.d_model, out_features=self.d_k, bias=False
        )
        self.fc_v = nn.Linear(
            in_features=self.d_model, out_features=self.d_v, bias=False
        )
        self.fc_o = nn.Linear(
            in_features=self.d_v, out_features=self.d_model, bias=False
        )

        self.softmax = nn.Softmax(dim=3)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        B: batch size
        S: source length — max length for source sequence
        T: target length — max length for target sequence

        If using self attention, S = T, but in general case S != T

        Args:
            query (torch.Tensor): Queries, tensor of size (B, T, d_model)
            key (torch.Tensor): Keys, tensor of size (B, S, d_model)
            value (torch.Tensor): Values, tensor of size (B, S, d_model)
            mask (torch.Tensor | None): Mask for Masked Self Attention.
                Tensor of size (B, S)

        Returns:
            torch.Tensor: Attention output, tensor of size (B, T, d_model)
            torch.Tensor: Attention weights, tensor of size (B, num_heads, T, S)
        """

        B = query.size()[0]
        T = query.size()[1]
        S = key.size()[1]

        # Q (B, T, d_k)
        # K (B, S, d_k)
        # V (B, S, d_v)
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # divide them to h attention heads

        # Q (B, num_heads, T, d_k_per_head)
        # K_t (B, num_heads, d_k_per_head, S)
        # V (B, num_heads, S, d_v_per_head)
        Q = Q.view(B, T, self.num_heads, self.d_k_per_head).permute(0, 2, 1, 3)
        K_t = K.view(B, S, self.num_heads, self.d_k_per_head).permute(0, 2, 3, 1)
        V = V.view(B, S, self.num_heads, self.d_v_per_head).permute(0, 2, 1, 3)

        # (B, num_heads, T, S)
        energy = torch.matmul(Q, K_t) * self.inv_sqrt_for_head

        if mask is not None:
            # if mask is not None => need to not look at certain positions

            # mask: (B, S)
            # mask: (B, 1, 1, S)
            mask = mask.unsqueeze(1).unsqueeze(1)

            # energy: (B, num_heads, T, S)
            energy = energy.masked_fill(mask == 0, -torch.inf)

        # (B, num_heads, T, S)
        a = self.softmax(energy)

        # (B, num_heads, T, d_v_per_head)
        attention_for_values = torch.matmul(a, V)

        # (B, T, d_v)
        concat = (
            attention_for_values.permute(0, 2, 1, 3)
            .reshape(B, T, self.d_v)
            .contiguous()
        )

        # (B, T, d_model)
        y = self.fc_o(concat)

        return y, a


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        """
        FeedForward

        FeedForward is a sequence of two Linear Layers
            with ReLu activation function between them.

        The first linear layer broadens the tensor,
            while the second one narrows it down to previous dimension.

        Args:
            d_model (int): dimension for input and output
            d_ff (int): dimension for inner linear layer
        """
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.fc_1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x (torch.Tensor): input tensor of size (B, max_length, d_model)

        Returns:
            torch.Tensor: output tensor of size (B, max_length, d_model)
        """
        y = self.fc_1(x)
        y = self.relu(y)
        y = self.fc_2(y)
        return y


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        d_k: int | None = None,
        d_v: int | None = None,
        p_dropout: float = 0,
    ) -> None:
        """
        Encoder layer of Transformer

        This implementation is slightly different from the one in original paper.
        This is Pre-LN Transformer Layer.

        For input x:
        x := x + Dropout1(MultiHeadAttention(LayerNorm1(x)))
        x := x + Dropout2(FeedForward(LayerNorm2(x)))

        Args:
            d_model (int): total dim size for input embeddings and resulting tensor
            num_heads (int): number of attention heads. Note that d_k must be divisible by num_heads
            d_ff (int): dim size for Feed Forward sublayer
            d_k (int): total dim size for Keys and Queries. Defaults to d_model
            d_v (int): total dim size for Values. Defaults to d_model
            p_dropout (float, optional): Dropout probability after each sublayer (MHA and FF). Defaults to 0.
        """
        super().__init__()

        if d_k is None:
            d_k = d_model
        if d_v is None:
            d_v = d_model

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.p_dropout = p_dropout

        self.layer_norm_attention = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(
            d_model=d_model, d_k=d_k, d_v=d_v, num_heads=num_heads
        )
        self.layer_norm_ff = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff)
        self.dropout_attention = nn.Dropout(p_dropout)
        self.dropout_ff = nn.Dropout(p_dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        For input x:
        x := x + Dropout1(MultiHeadAttention(LayerNorm1(x)))
        x := x + Dropout2(FeedForward(LayerNorm2(x)))

        Args:
            x (torch.Tensor): input of size (B, S, d_model)
            mask (Optional[torch.Tensor]): mask of size (B, S) where 0 means pad tokens. Defaults to None

        Returns:
            torch.Tensor: output of size (B, S, d_model)
            torch.Tensor: attention weights of size (B, S, S)
        """
        x_norm = self.layer_norm_attention(x)

        attention_output, attention_weights = self.mha(
            x_norm, x_norm, x_norm, mask=mask
        )
        x = x + self.dropout_attention(attention_output)

        x_norm = self.layer_norm_ff(x)
        x = x + self.dropout_ff(self.ff(x_norm))

        return x, attention_weights


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        n_layers: int,
        src_vocab_size: int,
        d_ff: int,
        d_k: int | None = None,
        d_v: int | None = None,
        p_dropout: float = 0,
        max_length: int = 50,
    ) -> None:
        """
        Transformer Encoder

        Args:
            d_model (int): total dim size for input embeddings and resulting tensor
            num_heads (int): number of attention heads. Note that d_k must be divisible by num_heads
            n_layers (int): number of encoder layers
            src_vocab_size (int): vocabulary size for input sequences, equals to number of embeddings
            d_ff (int): dim size for Feed Forward sublayer
            d_k (int): total dim size for Keys and Queries. Defaults to d_model
            d_v (int): total dim size for Values. Defaults to d_model
            p_dropout (float, optional): Dropout probability after embeddings and after each sublayer (MHA and FF). Defaults to 0.
            max_length (int, optional): Max length of an input. Defaults to 50.
        """
        super().__init__()

        if d_k is None:
            d_k = d_model
        if d_v is None:
            d_v = d_model

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = num_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.src_vocab_size = src_vocab_size
        self.p_dropout = p_dropout
        self.max_length = max_length

        self.embedding = nn.Embedding(
            num_embeddings=src_vocab_size, embedding_dim=d_model
        )
        self.positional_encoding = nn.Embedding(
            num_embeddings=max_length, embedding_dim=d_model
        )
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    d_k=d_k,
                    d_v=d_v,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    p_dropout=p_dropout,
                )
                for i in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Sequentially apply every layer

        Args:
            x (torch.Tensor): input of size (B, S)
            mask (Optional[torch.Tensor]): mask of size (B, S) where 0 means pad tokens. Defaults to None

        Returns:
            torch.Tensor: output of size (B, S, d_model)
            torch.Tensor: attention weights of size (B, S, S)
        """

        # TODO change docstring

        S = x.size()[1]

        assert S <= self.max_length, "Input too large"  # TODO maybe can extrapolate

        # (B, S, d_model)
        x = self.embedding(x)

        # (1, S)
        pos_encoding = torch.arange(S).unsqueeze(0).to(x.device)

        # (1, S, d_model)
        pos_encoding = self.positional_encoding(pos_encoding)

        # (B, S, d_model)
        x = self.dropout(x + pos_encoding)

        for layer in self.layers:
            x, attention_weights = layer(x, mask)

        return x, attention_weights


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        d_k: int | None = None,
        d_v: int | None = None,
        p_dropout: float = 0,
    ) -> None:
        """
        Decoder layer of Transformer

        This implementation is slightly different from the one in original paper.
        This is Pre-LN Transformer Layer.

        For input x:
        x := x + Dropout1(MultiHeadAttention(LayerNorm1(x)))
        x := x + Dropout2(MultiHeadAttention(LayerNorm1(2), encoder_outputs))
        x := x + Dropout3(FeedForward(LayerNorm3(x)))

        Args:
            d_model (int): total dim size for input embeddings and resulting tensor
            num_heads (int): number of attention heads. Note that d_k must be divisible by num_heads
            d_ff (int): dim size for Feed Forward sublayer
            d_k (int): total dim size for Keys and Queries. Defaults to d_model
            d_v (int): total dim size for Values. Defaults to d_model
            p_dropout (float, optional): Dropout probability after each sublayer (
                Masked MHA, MHA using Encoder outputs and FF). Defaults to 0.
        """
        super().__init__()  # TODO change docstring

        if d_k is None:
            d_k = d_model
        if d_v is None:
            d_v = d_model

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.p_dropout = p_dropout

        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.layer_norm_3 = nn.LayerNorm(d_model)

        self.masked_mha = MultiHeadAttention(
            d_model=d_model, d_k=d_k, d_v=d_v, num_heads=num_heads
        )
        self.mha_encoder = MultiHeadAttention(
            d_model=d_model, d_k=d_k, d_v=d_v, num_heads=num_heads
        )
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff)

        self.dropout_1 = nn.Dropout(p_dropout)
        self.dropout_2 = nn.Dropout(p_dropout)
        self.dropout_3 = nn.Dropout(p_dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output_norm: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        trg_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass

        For input x:
        x := x + Dropout1(MultiHeadAttention(LayerNorm1(x)))
        x := x + Dropout2(MultiHeadAttention(LayerNorm2(x), encoder_outputs))
        x := x + Dropout3(FeedForward(LayerNorm3(x)))

        Args:
            x (torch.Tensor): input of size (B, T, d_model)
            encoder_output_norm (torch.Tensor): Encoder output that was normalized through LayerNorm,
                tensor of size (B, S, d_model)
            src_mask (Optional[torch.Tensor]): mask of size (B, S). It is used to make attention calculation without padding
                tokens
            trg_mask (Optional[torch.Tensor]): mask of size (B, T). It is used to make attention calculation without padding
                tokens and future tokens

        Returns:
            torch.Tensor: output of size (B, T, d_model)
            torch.Tensor: Self Attention weights, (B, T, T)
            torch.Tensor: Attention using source sequence and encoder, (B, T, S)
        """
        x_norm = self.layer_norm_1(x)
        self_attention_out, self_attention_weights = self.masked_mha(
            x_norm, x_norm, x_norm, mask=trg_mask
        )
        x = x + self.dropout_1(self_attention_out)

        x_norm = self.layer_norm_2(x)
        encoder_attention_out, encoder_attention_weights = self.mha_encoder(
            x_norm, encoder_output_norm, encoder_output_norm, src_mask
        )
        x = x + self.dropout_2(encoder_attention_out)

        x_norm = self.layer_norm_3(x)
        x = x + self.dropout_3(self.ff(x_norm))

        return x, self_attention_weights, encoder_attention_weights


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        n_layers: int,
        trg_vocab_size: int,
        d_ff: int,
        d_k: int | None = None,
        d_v: int | None = None,
        p_dropout: float = 0,
        max_length: int = 50,
    ) -> None:
        """
        Transformer Encoder

        Args:
            d_model (int): total dim size for input embeddings and resulting tensor
            num_heads (int): number of attention heads. Note that d_k must be divisible by num_heads
            n_layers (int): number of encoder layers
            trg_vocab_size (int): vocabulary size for input and output sequences, equals to number of embeddings
            d_ff (int): dim size for Feed Forward sublayer
            d_k (int): total dim size for Keys and Queries. Defaults to d_model
            d_v (int): total dim size for Values. Defaults to d_model
            p_dropout (float, optional): Dropout probability after embeddings and after each sublayer (MHA and FF). Defaults to 0.
            max_length (int, optional): Max length of an input. Defaults to 50.
        """
        super().__init__()

        if d_k is None:
            d_k = d_model
        if d_v is None:
            d_v = d_model

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = num_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.trg_vocab_size = trg_vocab_size
        self.p_dropout = p_dropout
        self.max_length = max_length

        self.embedding = nn.Embedding(
            num_embeddings=trg_vocab_size, embedding_dim=d_model
        )
        self.positional_encoding = nn.Embedding(
            num_embeddings=max_length, embedding_dim=d_model
        )
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    d_k=d_k,
                    d_v=d_v,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    p_dropout=p_dropout,
                )
                for i in range(self.n_layers)
            ]
        )
        self.dropout = nn.Dropout(p=p_dropout)
        self.fc = nn.Linear(in_features=d_model, out_features=trg_vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output_norm: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        trg_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass

        For input x:
        x := x + Dropout1(MultiHeadAttention(LayerNorm1(x)))
        x := x + Dropout2(MultiHeadAttention(LayerNorm2(x), encoder_outputs))
        x := x + Dropout3(FeedForward(LayerNorm3(x)))

        Args:
            x (torch.Tensor): input of size (B, T, d_model)
            encoder_output_norm (torch.Tensor): Encoder output that was normalized through LayerNorm,
                tensor of size (B, S, d_model)
            src_mask (Optional[torch.Tensor]): mask of size (B, S). It is used to make attention calculation without padding
                tokens
            trg_mask (Optional[torch.Tensor]): mask of size (B, T). It is used to make attention calculation without padding
                tokens and future tokens

        Returns:
            torch.Tensor: output of size (B, T, trg_vocab_size)
            torch.Tensor: Self Attention weights, (B, T, T)
            torch.Tensor: Attention using source sequence and encoder, (B, T, S)
        """
        # TODO change docstring

        T = x.size()[1]

        assert T <= self.max_length, "Input too large"  # TODO maybe can extrapolate

        # (B, T, d_model)
        x = self.embedding(x)

        # (1, T)
        pos_encoding = torch.arange(T).unsqueeze(0).to(x.device)

        # (1, T, d_model)
        pos_encoding = self.positional_encoding(pos_encoding)

        # (B, T, d_model)
        x = self.dropout(x + pos_encoding)

        for layer in self.layers:
            x, self_attention, enc_attention = layer(
                x, encoder_output_norm, src_mask, trg_mask
            )

        # (B, T, trg_vocab_size)
        x = self.fc(x)

        return x, self_attention, enc_attention


class Transformer(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        n_layers_encoder: int,
        n_layers_decoder: int,
        src_vocab_size: int,
        trg_vocab_size: int,
        device: torch.device,
        pad_ind_src: int,
        pad_ind_trg: int,
        bos_ind: int,
        eos_ind: int,
        d_ff: int,
        d_k: int | None = None,
        d_v: int | None = None,
        p_dropout: float = 0,
        max_length: int = 50,
    ) -> None:
        """
        Transformer model

        It consists of Encoder with `n_layers_encoder` layers
        and Decoder with `n_layers_decoder` layers

        Args:
            d_model (int): _description_
            num_heads (int): _description_
            n_layers_encoder (int): _description_
            n_layers_decoder (int): _description_
            src_vocab_size (int): _description_
            trg_vocab_size (int): _description_
            device: torch.device
            pad_ind_encoder: int,
            pad_ind_decoder: int,
            bos_ind: int,
            eos_ind: int,
            d_ff (int): _description_
            d_k (int | None, optional): _description_. Defaults to None.
            d_v (int | None, optional): _description_. Defaults to None.
            p_dropout (float, optional): _description_. Defaults to 0.
            max_length (int, optional): _description_. Defaults to 50.
        """
        super().__init__()

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = num_heads
        self.pad_ind_src = pad_ind_src
        self.pad_ind_trg = pad_ind_trg
        self.bos_ind = bos_ind
        self.eos_ind = eos_ind
        self.device = device
        self.d_ff = d_ff
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.p_dropout = p_dropout
        self.max_length = max_length

        self.encoder = Encoder(
            d_model=d_model,
            num_heads=num_heads,
            n_layers=n_layers_encoder,
            src_vocab_size=src_vocab_size,
            d_ff=d_ff,
            d_k=d_k,
            d_v=d_v,
            p_dropout=p_dropout,
            max_length=max_length,
        ).to(device)

        self.encoder_layer_norm = nn.LayerNorm(normalized_shape=d_model).to(device)

        self.decoder = Decoder(
            d_model=d_model,
            num_heads=num_heads,
            n_layers=n_layers_encoder,
            trg_vocab_size=trg_vocab_size,
            d_ff=d_ff,
            d_k=d_k,
            d_v=d_v,
            p_dropout=p_dropout,
            max_length=max_length,
        ).to(device)

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        src_mask: torch.Tensor,
        trg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for Transformer

        Args:
            src (torch.Tensor): source sequence, tensor of size (B, S)
            trg (torch.Tensor): target sequence, tensor of size (B, T)

        Returns:
            torch.Tensor: logits of predictions, tensor of size (B, T, trg_vocab_size)
        """
        # src_mask = self._create_src_mask(src)
        # trg_mask = self._create_trg_mask(trg)

        # TODO change docstring, remove _create_mask

        # (B, S, d_model)
        encoder_output, encoder_attention = self.encoder(src, src_mask)

        # (B, S, d_model)
        encoder_output_norm = self.encoder_layer_norm(encoder_output)

        # (B, T, trg_vocab_size)
        decoder_output, decoder_self_attention, decoder_encoder_attention = (
            self.decoder(trg, encoder_output_norm, src_mask, trg_mask)
        )
        return decoder_output

    @torch.inference_mode()
    def inference(self, src: torch.Tensor) -> torch.Tensor:
        """
        Perform computation of output for src tensor until reaching either `eos_ind` or `max_length`

        Args:
            src (torch.Tensor): input tensor of size (S)

        Returns:
            torch.Tensor: output tensor of size (T), T <= max_length
        """

        # src: (1, S)
        src = src.reshape(1, -1)

        # src_mask: (1, S)
        src_mask = torch.ones_like(src).to(src.device)

        # (1, S, d_model)
        encoder_output, encoder_attention = self.encoder(src, src_mask)

        # (1, S, d_model)
        encoder_output_norm = self.encoder_layer_norm(encoder_output)

        # (1, 1), later (1, T)
        decoder_input = torch.tensor([self.bos_ind]).unsqueeze(0).to(src.device)

        # TODO unoptimized
        while (
            decoder_input[0][-1] != self.eos_ind
            and decoder_input.nelement() < self.max_length
        ):
            # (1, T, trg_size)
            decoder_output, decoder_self_attention, decoder_encoder_attention = (
                self.decoder(
                    decoder_input,
                    encoder_output_norm,
                    src_mask,
                    torch.ones_like(decoder_input),
                )
            )

            # (1, 1)
            next_token = torch.argmax(decoder_output[0, -1]).reshape(1, 1)

            decoder_input = torch.concat((decoder_input, next_token), dim=1)

        # (T)
        return decoder_input.squeeze(0)

    def _create_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Create src mask for input

        Args:
            src (torch.Tensor): source sequence, tensor of size (B, S)

        Returns:
            torch.Tensor: mask of 1 and 0, tensor of size (B, S)
        """
        return (src != self.pad_ind_src).int().to(self.device)

    def _create_trg_mask(self, trg: torch.Tensor) -> torch.Tensor:
        """
        Create target

        Args:
            trg (torch.Tensor): target sequence, tensor of size (B, T)

        Returns:
            torch.Tensor: mask of 1 and 0, tensor of size (B, T)
        """
        mask = (trg != self.pad_ind_trg).int().to(self.device)
        mask = torch.tril(mask)
        return mask
