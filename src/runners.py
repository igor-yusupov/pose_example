# from catalyst.dl import Runner
# from catalyst import metrics
from catalyst import dl, metrics


class OpenPoseRunner(dl.Runner):
    def __init__(self):
        super().__init__()

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveMetric(compute_on_call=False)
            for key in ["loss"]
        }
        for i in range(6):
            for key in [f'train_paf_loss_stage{i}',
                        f'train_heatmap_loss_stage{i}',
                        f'val_paf_loss_stage{i}',
                        f'val_heatmap_loss_stage{i}']:
                self.meters[key] = metrics.AdditiveMetric(
                    compute_on_call=False)

        if self.is_train_loader:
            if self.global_epoch_step == 2:
                for param in self.model.base.vgg_base.parameters():
                    param.requires_grad = True
                self.optimizer.add_param_group(
                    {'params': [*self.model.base.vgg_base.parameters()],
                     'lr': self.optimizer.param_groups[-1]["lr"] / 4})

            if self.global_epoch_step == 10 or self.global_epoch_step == 20:
                self.lr_schedule()

    def handle_batch(self, batch):
        imgs, pafs, heatmaps, ignore_mask = (
            batch['img'], batch['pafs'], batch['heatmaps'],
            batch['ignore_mask'])

        pafs_ys, heatmaps_ys = self.model(imgs)

        loss, paf_loss_log, heatmap_loss_log = self.criterion(
            pafs_ys, heatmaps_ys, pafs, heatmaps, ignore_mask)

        self.batch_metrics.update(
            {"loss": loss, "lr": self.optimizer.param_groups[-1]["lr"]}
        )

        prefix = 'train' if self.is_train_loader else 'val'
        for stage, (paf_loss, heatmap_loss) in enumerate(
                zip(paf_loss_log, heatmap_loss_log)):
            self.batch_metrics.update(
                {f"{prefix}_paf_loss_stage{stage}": paf_loss,
                 f"{prefix}_heatmap_loss_stage{stage}": heatmap_loss}
            )

        for key in ["loss"]:
            self.meters[key].update(self.batch_metrics[key].item(),
                                    self.batch_size)
        for i in range(6):
            for key in [f'{prefix}_paf_loss_stage{i}',
                        f'{prefix}_heatmap_loss_stage{i}']:
                self.meters[key].update(self.batch_metrics[key].item(),
                                        self.batch_size)

        if self.is_train_loader:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def on_loader_end(self, runner):
        for key in ["loss"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]

        prefix = 'train' if self.is_train_loader else 'val'
        for i in range(6):
            for key in [f'{prefix}_paf_loss_stage{i}',
                        f'{prefix}_heatmap_loss_stage{i}']:
                self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)

    def lr_schedule(self):
        for params in self.optimizer.param_groups:
            params['lr'] /= 10.
        print(self.optimizer)


# class OpenPoseRunner(Runner):
#     def __init__(self, device):
#         super().__init__()
#
#     def handle_batch(self, batch):
#         out = self.model(batch)
#
#         loss_dict = self.criterion(out)
#         loss = loss_dict['loss']
#
#         self.batch_metrics.update(
#             {"loss": loss, "lr": self.optimizer.param_groups[-1]["lr"]}
#         )
#
#         if self.is_train_loader:
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()


def get_runner(config: dict):
    required_runner = config.get("runner")
    name = required_runner["name"]

    runner = globals().get(name)

    return runner()
