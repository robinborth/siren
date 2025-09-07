import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm

from utils import gradient_fn, sdf_loss, write_sdf_summary


def train_custom(model, train_dataloader, epochs, lr, model_dir):
    summaries_dir = Path(model_dir) / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = Path(model_dir) / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(summaries_dir)
    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    pbar = tqdm(total=len(train_dataloader) * epochs)
    for epoch in range(epochs):
        for batch in train_dataloader:
            batch = {key: value.cuda() for key, value in batch.items()}

            sdf_in = batch["pcd_points"].clone().detach().requires_grad_(True)
            sdf_out = model(sdf_in)
            sdf_grad = gradient_fn(sdf_out, sdf_in)

            eikonal_in = batch["eikonal_points"].clone().detach().requires_grad_(True)
            eikonal_out = model(eikonal_in)
            eikonal_grad = gradient_fn(eikonal_out, eikonal_in)

            sdf_constraint = sdf_out
            normal_constraint = 1 - F.cosine_similarity(
                sdf_grad, batch["pcd_normals"], dim=-1
            )
            inter_constraint = torch.exp(-1e2 * torch.abs(eikonal_out))

            grad_constraint = torch.cat(
                [
                    torch.abs(sdf_grad.norm(dim=-1) - 1),
                    torch.abs(eikonal_grad.norm(dim=-1) - 1),
                ]
            )

            losses = {
                "sdf": torch.abs(sdf_constraint).mean() * 3e3,
                "inter": inter_constraint.mean() * 1e2,
                "normal_constraint": normal_constraint.mean() * 1e2,
                "grad_constraint": grad_constraint.mean() * 1e2,
            }
            total_loss = 0.0
            for _, loss in losses.items():
                single_loss = loss.mean()
                total_loss += single_loss

            optim.zero_grad()
            total_loss.backward()

            optim.step()

            pbar.update(1)
            pbar.set_postfix({"loss": total_loss.item()})
            if epoch % 100 == 0:
                print({k: round(v.item(), 3) for k, v in losses.items()})


def train_simple(model, train_dataloader, epochs, lr, clip_grad=False, **kwargs):
    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    pbar = tqdm(total=len(train_dataloader) * epochs)
    for epoch in range(epochs):
        for model_input, gt in train_dataloader:
            model_input = {key: value.cuda() for key, value in model_input.items()}
            gt = {key: value.cuda() for key, value in gt.items()}

            model_output = model(model_input)
            losses = sdf_loss(model_output, gt)
            total_loss = 0.0
            for _, loss in losses.items():
                single_loss = loss.mean()
                total_loss += single_loss

            optim.zero_grad()
            total_loss.backward()

            optim.step()

            pbar.update(1)
            pbar.set_postfix({"loss": total_loss.item()})
            if epoch % 100 == 0:
                print({k: round(v.item(), 3) for k, v in losses.items()})


def train(
    model,
    train_dataloader,
    epochs,
    lr,
    steps_til_summary,
    epochs_til_checkpoint,
    model_dir,
    clip_grad=False,
):
    optim = torch.optim.Adam(lr=lr, params=model.parameters())

    summaries_dir = Path(model_dir) / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = Path(model_dir) / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch:
                torch.save(
                    model.state_dict(),
                    os.path.join(checkpoints_dir, "model_epoch_%04d.pth" % epoch),
                )
                np.savetxt(
                    os.path.join(
                        checkpoints_dir, "train_losses_epoch_%04d.txt" % epoch
                    ),
                    np.array(train_losses),
                )

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()

                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                model_output = model(model_input)
                losses = sdf_loss(model_output, gt)

                train_loss = 0.0
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()
                    writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                train_losses.append(train_loss.item())
                writer.add_scalar("total_train_loss", train_loss, total_steps)

                if not total_steps % steps_til_summary:
                    torch.save(
                        model.state_dict(),
                        os.path.join(checkpoints_dir, "model_current.pth"),
                    )
                    write_sdf_summary(
                        model,
                        model_input,
                        gt,
                        model_output,
                        writer,
                        total_steps,
                    )

                optim.zero_grad()
                train_loss.backward()

                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optim.step()
                pbar.update(1)

                if not total_steps % steps_til_summary:
                    tqdm.write(
                        "Epoch %d, Total loss %0.6f, iteration time %0.6f"
                        % (epoch, train_loss, time.time() - start_time)
                    )

                total_steps += 1

        torch.save(model.state_dict(), os.path.join(checkpoints_dir, "model_final.pth"))
        np.savetxt(
            os.path.join(checkpoints_dir, "train_losses_final.txt"),
            np.array(train_losses),
        )
