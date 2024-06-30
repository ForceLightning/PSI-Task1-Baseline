from __future__ import annotations
from typing import Any, overload

import torch
from torch import nn
from torch.optim import Adam, AdamW, Optimizer, SGD
from torch.optim.lr_scheduler import (
    ExponentialLR,
    LRScheduler,
    OneCycleLR,
    ReduceLROnPlateau,
)
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
from transformers.modeling_outputs import (
    SampleTSPredictionOutput,
    Seq2SeqTSModelOutput,
    Seq2SeqTSPredictionOutput,
)
from typing_extensions import override

from data.custom_dataset import T_intentBatch
from models.base_model import IConstructOptimizer
from utils.args import DefaultArguments, ModelOpts
from utils.cuda import *


class TransformerTrajBbox(nn.Module, IConstructOptimizer):
    """A transformer-based architecture to predict the trajectory of pedestrians
    from their past bounding boxes.

    :ivar DefaultArguments args: Training arguments.
    :ivar ModelOpts model_opts: Model configuration options.
    :ivar TimeSeriesTransformerConfig config: Config for the transformer model.
    :ivar TimeSeriesTransformerForPrediction transformer: The transformer model.
    """

    def __init__(
        self, args: DefaultArguments, config: TimeSeriesTransformerConfig
    ) -> None:
        """Initialises the model.

        :param DefaultArguments args: Training arguments.
        :param TimeSeriesTransformerConfig config: Config for the transformer model.
        """
        super().__init__()
        self.args = args
        self.model_opts: ModelOpts = args.model_configs
        self.config = config
        self.transformer = TimeSeriesTransformerForPrediction(self.config)

    @overload
    def forward(
        self, x: T_intentBatch
    ) -> Seq2SeqTSPredictionOutput | tuple[torch.Tensor, ...]: ...

    @overload
    def forward(
        self, x: tuple[torch.Tensor, torch.Tensor]
    ) -> Seq2SeqTSPredictionOutput | tuple[torch.Tensor, ...]: ...

    @override
    def forward(
        self, x: T_intentBatch | tuple[torch.Tensor, torch.Tensor]
    ) -> Seq2SeqTSPredictionOutput | tuple[torch.Tensor, ...]:
        """Forward pass for training the model. See :py:meth:`generate` to generate
            inferences.

        :param x: Input data, dictionary if using custom dataset, or tuple if using
            :py:mod:`lr_finder`.
        :type x: T_intentBatch or tuple[torch.Tensor, torch.Tensor]
        :returns: Transformer prediction output.
        :rtype: Seq2SeqTSPredictionOutput or tuple[torch.Tensor, ...]
        """
        # Inputs to the transformer
        past_values: torch.Tensor
        past_time_features: torch.Tensor
        future_values: torch.Tensor
        future_time_features: torch.Tensor
        match x:
            case dict():  # type: ignore[reportUnnecessaryCondition]
                past_values = x["bboxes"][:, : self.args.observe_length, :].type(
                    FloatTensor
                )
                past_time_features = x["total_frames"][
                    :, : self.args.observe_length
                ].type(FloatTensor)
                future_values = x["bboxes"][:, self.args.observe_length :, :].type(
                    FloatTensor
                )
                future_time_features = x["total_frames"][
                    :, self.args.observe_length :
                ].type(FloatTensor)
            case tuple():
                past_values = x[0][:, : self.args.observe_length, :].type(FloatTensor)
                future_values = x[0][:, self.args.observe_length :, :].type(FloatTensor)
                past_time_features = x[1][:, : self.args.observe_length].type(
                    FloatTensor
                )
                future_time_features = x[1][:, self.args.observe_length :].type(
                    FloatTensor
                )
            case _:
                past_values = x[:, : self.args.observe_length, :].type(FloatTensor)
                future_values = x[:, self.args.observe_length :, :].type(FloatTensor)
                past_time_features = x[:, : self.args.observe_length].type(FloatTensor)
                future_time_features = x[:, self.args.observe_length :].type(
                    FloatTensor
                )

        past_time_features = past_time_features.unsqueeze(2)
        future_time_features = future_time_features.unsqueeze(2)
        bs, ts, _ = past_values.shape
        past_observed_mask: torch.Tensor = torch.ones_like(
            past_values, dtype=torch.bool
        )
        outputs: Seq2SeqTSPredictionOutput = self.transformer(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
            future_values=future_values,
            future_time_features=future_time_features,
        )

        return outputs

    @overload
    def generate(self, x: T_intentBatch) -> torch.Tensor: ...

    @overload
    def generate(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor: ...

    def generate(
        self, x: T_intentBatch | tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        past_values: torch.Tensor
        past_time_features: torch.Tensor
        future_time_features: torch.Tensor
        match x:
            case dict():  # type: ignore[reportUnnecessaryCondition]
                past_values = x["bboxes"][:, : self.args.observe_length, :].type(
                    FloatTensor
                )
                past_time_features = x["total_frames"][
                    :, : self.args.observe_length
                ].type(FloatTensor)
                future_time_features = x["total_frames"][
                    :, self.args.observe_length :
                ].type(FloatTensor)
            case tuple():
                past_values = x[0][:, : self.args.observe_length, :].type(FloatTensor)
                past_time_features = x[1][:, : self.args.observe_length].type(
                    FloatTensor
                )
                future_time_features = x[1][:, self.args.observe_length :].type(
                    FloatTensor
                )
            case _:
                past_values = x[:, : self.args.observe_length, :].type(FloatTensor)
                past_time_features = x[:, : self.args.observe_length].type(FloatTensor)
                future_time_features = x[:, self.args.observe_length :].type(
                    FloatTensor
                )

        past_time_features = past_time_features.unsqueeze(2)
        future_time_features = future_time_features.unsqueeze(2)
        bs, ts, _ = past_values.shape
        past_observed_mask: torch.Tensor = torch.ones_like(
            past_values, dtype=torch.bool
        )
        outputs: SampleTSPredictionOutput = self.transformer.generate(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
            future_time_features=future_time_features,
        )

        preds = outputs.sequences.mean(dim=1)
        return preds

    @override
    def build_optimizer(self, args: DefaultArguments) -> tuple[Optimizer, LRScheduler]:
        learning_rate = args.lr

        opt_eps = self.model_opts.get("opt_eps", 1e-4)
        opt_wd = self.model_opts.get("opt_wd", 1e-2)
        opt_name = self.model_opts.get("optimizer", "Adam")
        opt_mom = self.model_opts.get("momentum", 0.9)

        optimizer: Optimizer
        params = self.transformer.parameters()
        match opt_name:
            case "AdamW":
                optimizer = AdamW(
                    params,
                    lr=learning_rate,
                    eps=opt_eps,
                    weight_decay=opt_wd,
                    fused=CUDA,
                )

            case "SGD":
                optimizer = SGD(params, lr=learning_rate, momentum=opt_mom, fused=CUDA)

            case _:
                optimizer = Adam(
                    params,
                    lr=learning_rate,
                    eps=opt_eps,
                    weight_decay=opt_wd,
                    fused=CUDA,
                )

        for opt_param_grp in optimizer.param_groups:
            opt_param_grp["lr0"] = opt_param_grp["lr"]

        sched_name = self.model_opts.get("scheduler", "ExponentialLR")
        scheduler: LRScheduler
        match sched_name:
            case "OneCycleLR":
                sched_div_factor = self.model_opts.get("scheduler_div_factor", 10)
                sched_pct_start = self.model_opts.get("scheduler_pct_start", 0.2)

                scheduler = OneCycleLR(
                    optimizer,
                    learning_rate,
                    epochs=args.epochs,
                    steps_per_epoch=args.steps_per_epoch,
                    div_factor=sched_div_factor,
                    pct_start=sched_pct_start,
                )

            case "ReduceLROnPlateau":
                sched_factor = self.model_opts.get("scheduler_factor", 10**0.5 * 0.1)
                sched_threshold = self.model_opts.get("scheduler_threshold", 1e-2)

                scheduler = ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=sched_factor,
                    threshold=sched_threshold,
                )

            case _:
                sched_gamma = self.model_opts.get("scheduler_gamma", 0.9)
                scheduler = ExponentialLR(optimizer, gamma=sched_gamma)

        return optimizer, scheduler


class TransformerTrajBboxPose(TransformerTrajBbox):
    @overload
    def forward(
        self,
        x: T_intentBatch,
        output_hidden_states: bool | None = None,
    ) -> Seq2SeqTSPredictionOutput | tuple[torch.Tensor, ...]: ...

    @overload
    def forward(
        self,
        x: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        output_hidden_states: bool | None = None,
    ) -> Seq2SeqTSPredictionOutput | tuple[torch.Tensor, ...]: ...

    @override
    def forward(
        self,
        x: T_intentBatch | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        output_hidden_states: bool | None = None,
    ) -> Seq2SeqTSPredictionOutput | tuple[torch.Tensor, ...]:
        """Forward pass for training the model. See :py:meth:`generate` to generate
            inferences.

        :param x: Input data, dictionary if using custom dataset, or tuple if using
            :py:mod:`PSI_Intent_Prediction.utils.lr_finder`.
        :type x: T_intentBatch or tuple[torch.Tensor, torch.Tensor]
        :returns: Transformer prediction output.
        :rtype: Seq2SeqTSPredictionOutput or tuple[torch.Tensor, ...]
        """
        # Inputs to the transformer
        past_values: torch.Tensor
        past_frames: torch.Tensor
        past_poses: torch.Tensor
        future_values: torch.Tensor
        future_frames: torch.Tensor
        future_poses: torch.Tensor
        match x:
            case dict():  # type: ignore[reportUnnecessaryCondition]
                past_values = x["bboxes"][:, : self.args.observe_length, :].type(
                    FloatTensor
                )
                future_values = x["bboxes"][:, self.args.observe_length :, :].type(
                    FloatTensor
                )
                past_frames = x["total_frames"][:, : self.args.observe_length].type(
                    FloatTensor
                )
                future_frames = x["total_frames"][:, self.args.observe_length :].type(
                    FloatTensor
                )
                past_poses = x["pose"][:, : self.args.observe_length, :, :].type(
                    FloatTensor
                )
                future_poses = x["pose"][:, self.args.observe_length :, :, :].type(
                    FloatTensor
                )

            case tuple():
                past_values = x[0][:, : self.args.observe_length, :].type(FloatTensor)
                future_values = x[0][:, self.args.observe_length :, :].type(FloatTensor)
                past_frames = x[1][:, : self.args.observe_length].type(FloatTensor)
                future_frames = x[1][:, self.args.observe_length :].type(FloatTensor)
                past_poses = x[2][:, : self.args.observe_length, :].type(FloatTensor)
                future_poses = x[2][:, self.args.observe_length :, :].type(FloatTensor)

        past_frames = past_frames.unsqueeze(2)
        future_frames = future_frames.unsqueeze(2)

        past_poses = past_poses.reshape(-1, self.args.observe_length, 34)
        future_poses = future_poses.reshape(-1, self.args.predict_length, 34)

        past_time_features: torch.Tensor = torch.cat((past_frames, past_poses), dim=2)
        future_time_features: torch.Tensor = torch.cat(
            (future_frames, future_poses), dim=2
        )

        past_observed_mask: torch.Tensor = torch.ones_like(
            past_values, dtype=torch.bool
        )
        outputs: Seq2SeqTSPredictionOutput = self.transformer(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
            future_values=future_values,
            future_time_features=future_time_features,
            output_hidden_states=output_hidden_states,
        )

        return outputs

    @overload
    def generate(self, x: T_intentBatch) -> torch.Tensor: ...

    @overload
    def generate(
        self, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor: ...

    @override
    @torch.no_grad()
    def generate(
        self, x: T_intentBatch | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        past_values: torch.Tensor
        past_frames: torch.Tensor
        past_poses: torch.Tensor
        future_frames: torch.Tensor
        future_poses: torch.Tensor
        match x:
            case dict():  # type: ignore[reportUnnecessaryCondition]
                past_values = x["bboxes"][:, : self.args.observe_length, :].type(
                    FloatTensor
                )
                past_frames = x["total_frames"][:, : self.args.observe_length].type(
                    FloatTensor
                )
                future_frames = x["total_frames"][:, self.args.observe_length :].type(
                    FloatTensor
                )
                past_poses = x["pose"][:, : self.args.observe_length, :, :].type(
                    FloatTensor
                )
                future_poses = x["pose"][:, self.args.observe_length :, :, :].type(
                    FloatTensor
                )

            case tuple():
                past_values = x[0][:, : self.args.observe_length, :].type(FloatTensor)
                future_values = x[0][:, self.args.observe_length :, :].type(FloatTensor)

                past_frames = x[1][:, : self.args.observe_length].type(FloatTensor)
                future_frames = x[1][:, self.args.observe_length :].type(FloatTensor)

                past_poses = x[2][:, : self.args.observe_length, :, :].type(FloatTensor)
                future_poses = x[2][:, self.args.observe_length :, :, :].type(
                    FloatTensor
                )

        past_frames = past_frames.unsqueeze(2)
        future_frames = future_frames.unsqueeze(2)

        past_poses = past_poses.reshape(-1, self.args.observe_length, 34)
        future_poses = future_poses.reshape(-1, self.args.predict_length, 34)

        past_time_features: torch.Tensor = torch.cat((past_frames, past_poses), dim=2)
        future_time_features: torch.Tensor = torch.cat(
            (future_frames, future_poses), dim=2
        )

        past_observed_mask: torch.Tensor = torch.ones_like(
            past_values, dtype=torch.bool
        )
        outputs: SampleTSPredictionOutput = self.transformer.generate(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
            future_time_features=future_time_features,
        )

        preds = outputs.sequences.mean(dim=1)
        return preds


class TransformerTrajIntentBboxPose(TransformerTrajBboxPose, IConstructOptimizer):
    def __init__(
        self, args: DefaultArguments, config: TimeSeriesTransformerConfig
    ) -> None:
        super().__init__(args, config)

        self.intent_loss: nn.Module
        match self.model_opts.get("intent_loss", "BCEWithLogitsLoss"):
            case "MSELoss":
                self.intent_loss = nn.MSELoss()
            case "BCELoss":
                self.intent_loss = nn.BCELoss()
            case _:
                self.intent_loss = nn.BCEWithLogitsLoss()

        self.intent_activation: nn.Module
        match self.model_opts.get("intent_activation", None):
            case "ReLU":
                self.intent_activation = nn.ReLU()
            case "Sigmoid":
                self.intent_activation = nn.Sigmoid()
            case _:
                self.intent_activation = nn.Identity()

        # TODO: Maybe turn into TCN?
        self.intent_head = nn.Sequential(
            nn.Linear(self.config.d_model, 64),
            nn.Mish(),
            nn.Linear(64, 32),
            nn.Mish(),
            nn.Linear(32, 1),
        )

    @overload
    def forward(
        self, x: T_intentBatch
    ) -> Seq2SeqTSPredictionOutput | tuple[torch.Tensor, ...]: ...

    @overload
    def forward(
        self, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Seq2SeqTSPredictionOutput | tuple[torch.Tensor, ...]: ...

    @override
    def forward(
        self,
        x: (
            T_intentBatch
            | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ),
    ) -> Seq2SeqTSPredictionOutput | tuple[torch.Tensor, ...]:
        intent: torch.Tensor
        outputs: Seq2SeqTSPredictionOutput | tuple[torch.Tensor, ...]
        match x:
            case tuple():
                intent = (
                    x[3][:, self.args.observe_length :].type(FloatTensor).to(DEVICE)
                )
                outputs = super().forward(x[:3], output_hidden_states=True)
            case _:
                intent = (
                    x["intention_binary"][:, self.args.observe_length :]
                    .type(LongTensor)
                    .to(DEVICE)
                )
                outputs = super().forward(x, output_hidden_states=True)

        last_hidden_state = outputs.decoder_hidden_states[-1]
        intent_input = last_hidden_state.reshape(
            -1, self.args.predict_length, self.config.d_model
        )
        intent_logits = self.intent_head(intent_input)
        intent_probs = self.intent_activation(intent_logits).squeeze()

        intent_loss: torch.FloatTensor = self.intent_loss(intent_probs, intent)
        match outputs:
            case Seq2SeqTSPredictionOutput():
                outputs.loss += intent_loss
            case (prediction_loss, output):
                outputs = (prediction_loss + intent_loss,) + output

        return outputs

    @overload
    def generate(self, x: T_intentBatch) -> tuple[torch.Tensor, torch.Tensor]: ...

    @overload
    def generate(
        self, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @override
    @torch.no_grad()
    def generate(
        self,
        x: T_intentBatch | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        past_values: torch.Tensor
        past_frames: torch.Tensor
        past_poses: torch.Tensor
        future_frames: torch.Tensor
        future_poses: torch.Tensor
        match x:
            case dict():  # type: ignore[reportUnnecessaryCondition]
                past_values = x["bboxes"][:, : self.args.observe_length, :].type(
                    FloatTensor
                )
                past_frames = x["total_frames"][:, : self.args.observe_length].type(
                    FloatTensor
                )
                future_frames = x["total_frames"][:, self.args.observe_length :].type(
                    FloatTensor
                )
                past_poses = x["pose"][:, : self.args.observe_length, :, :].type(
                    FloatTensor
                )
                future_poses = x["pose"][:, self.args.observe_length :, :, :].type(
                    FloatTensor
                )

            case tuple():
                past_values = x[0][:, : self.args.observe_length, :].type(FloatTensor)
                future_values = x[0][:, self.args.observe_length :, :].type(FloatTensor)

                past_frames = x[1][:, : self.args.observe_length].type(FloatTensor)
                future_frames = x[1][:, self.args.observe_length :].type(FloatTensor)

                past_poses = x[2][:, : self.args.observe_length, :, :].type(FloatTensor)
                future_poses = x[2][:, self.args.observe_length :, :, :].type(
                    FloatTensor
                )

        past_frames = past_frames.unsqueeze(2)
        future_frames = future_frames.unsqueeze(2)

        past_poses = past_poses.reshape(-1, self.args.observe_length, 34)
        future_poses = future_poses.reshape(-1, self.args.predict_length, 34)

        past_time_features: torch.Tensor = torch.cat((past_frames, past_poses), dim=2)
        future_time_features: torch.Tensor = torch.cat(
            (future_frames, future_poses), dim=2
        )

        past_observed_mask: torch.Tensor = torch.ones_like(
            past_values, dtype=torch.bool
        )
        outputs: Seq2SeqTSModelOutput = self.transformer(
            static_categorical_features=None,
            static_real_features=None,
            past_time_features=past_time_features,
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            future_time_features=future_time_features,
            future_values=None,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
        )

        traj_full_preds, dec_last_hidden = self._generate_with_last_hidden_state(
            outputs
        )

        traj_preds = traj_full_preds.sequences.mean(dim=1)

        intent_logits: torch.Tensor = self.intent_head(dec_last_hidden)
        intent_probs: torch.Tensor = self.intent_activation(intent_logits).squeeze()

        return traj_preds, intent_probs

    @torch.no_grad()
    def _generate_with_last_hidden_state(
        self, outputs: Seq2SeqTSModelOutput
    ) -> tuple[SampleTSPredictionOutput, torch.Tensor]:
        decoder = self.transformer.model.get_decoder()
        enc_last_hidden = outputs.encoder_last_hidden_state
        loc = outputs.loc
        scale = outputs.scale
        static_feat = outputs.static_features

        num_parallel_samples = self.config.num_parallel_samples
        repeated_loc = loc.repeat_interleave(repeats=num_parallel_samples, dim=0)
        repeated_scale = scale.repeat_interleave(repeats=num_parallel_samples, dim=0)

        repeated_past_values = (
            past_values.repeat_interleave(repeats=num_parallel_samples, dim=0)
            - repeated_loc
        ) / repeated_scale

        expanded_static_feat = static_feat.unsqueeze(1).expand(
            -1, future_time_features.shape[1], -1
        )
        features = torch.cat((expanded_static_feat, future_time_features), dim=-1)
        repeated_features = features.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        )

        repeated_enc_last_hidden = enc_last_hidden.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        )

        future_samples = []

        # greedy decoding
        dec_last_hidden: torch.Tensor
        for k in range(self.config.prediction_length):
            lagged_sequence = self.model.get_lagged_subsequences(
                sequence=repeated_past_values,
                subsequences_length=1 + k,
                shift=1,
            )

            lags_shape = lagged_sequence.shape
            reshaped_lagged_sequence = lagged_sequence.reshape(
                lags_shape[0], lags_shape[1], -1
            )

            decoder_input = torch.cat(
                (reshaped_lagged_sequence, repeated_features[:, : k + 1]), dim=-1
            )

            dec_output = decoder(
                inputs_embeds=decoder_input,
                encoder_hidden_states=repeated_enc_last_hidden,
            )
            dec_last_hidden = dec_output.last_hidden_state

            params = self.parameter_projection(dec_last_hidden[:, -1:])
            distr = self.output_distribution(
                params, loc=repeated_loc, scale=repeated_scale
            )
            next_sample = distr.sample()

            repeated_past_values = torch.cat(
                (repeated_past_values, (next_sample - repeated_loc) / repeated_scale),
                dim=1,
            )
            future_samples.append(next_sample)

        concat_future_samples = torch.cat(future_samples, dim=1)

        pred_output = SampleTSPredictionOutput(
            sequences=concat_future_samples.reshape(
                -1, self.config.prediction_length, self.config.target_dim
            )
        )

        return pred_output, dec_last_hidden
