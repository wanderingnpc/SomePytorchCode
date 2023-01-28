import os
import math
import logging
from tqdm.auto import tqdm
import torch
from torch.optim import AdamW
from transformers import AutoModelForMaskedLM, get_scheduler
from accelerate import Accelerator


def training_mlm(config):
    if config.check_point:
        model = AutoModelForMaskedLM.from_pretrained(config.check_point)
    else:
        model = AutoModelForMaskedLM.from_pretrained(config.model_name)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, config.train_dataloader, config.eval_dataloader
    )

    num_steps_per_epoch = len(train_dataloader)
    num_training_steps = config.num_train_epochs * num_steps_per_epoch // config.gradient_accumulation_steps

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_main_process)
    i_step = config.start_step  # model update step
    accumulation_step = 1  # gradient accumulation step
    train_losses = []

    for epoch in range(config.num_train_epochs):
        for batch in train_dataloader:
            # Training
            model.train()
            outputs = model(**batch)
            train_loss = outputs.loss

            train_losses.append(accelerator.gather(train_loss.repeat(config.train_batch_size)))

            train_loss = train_loss / config.gradient_accumulation_steps
            accelerator.backward(train_loss)

            if accumulation_step % config.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)

                # Evaluation
                if i_step % config.eval_per_step == 0:
                    model.eval()
                    eval_losses = []
                    for step, batch_eval in enumerate(eval_dataloader):
                        with torch.no_grad():
                            outputs = model(**batch_eval)

                        eval_loss = outputs.loss
                        eval_losses.append(accelerator.gather(eval_loss.repeat(config.eval_batch_size)))

                    eval_losses = torch.cat(eval_losses)
                    eval_losses = eval_losses[: len(config.eval_dataloader)]
                    try:
                        eval_perplexity = math.exp(torch.mean(eval_losses))
                    except OverflowError:
                        eval_perplexity = float("inf")

                    train_losses = torch.cat(train_losses)
                    train_losses = train_losses[: config.eval_per_step]
                    try:
                        train_perplexity = math.exp(torch.mean(train_losses))
                    except OverflowError:
                        train_perplexity = float("inf")

                    if accelerator.is_main_process:
                        logging.info(
                            f"Step {i_step}: Train Perplexity: {train_perplexity}, Val Perplexity: {eval_perplexity}\n")

                    train_losses = []
                    model.train()

                # Save model
                if i_step % config.save_per_step == 0:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(os.path.join(config.model_save_path, str(i_step)),
                                                    save_function=accelerator.save)

                i_step += 1

            accumulation_step += 1


def training_mlm_cl(config):
    if config.check_point:
        model = AutoModelForMaskedLM.from_pretrained(config.check_point)
    else:
        model = AutoModelForMaskedLM.from_pretrained(config.model_name)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, config.train_dataloader, config.eval_dataloader
    )
    num_steps_per_epoch = len(train_dataloader)
    num_training_steps = config.num_train_epochs * num_steps_per_epoch // config.gradient_accumulation_steps

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_main_process)
    i_step = config.start_step  # model update step
    accumulation_step = 1  # gradient accumulation step
    train_losses = []

    for epoch in range(config.num_train_epochs):
        for batch in train_dataloader:
            # Training
            model.train()

            sent1_outputs = model(input_ids=batch.input_ids1,
                                  token_type_ids=batch.token_type_ids1,
                                  attention_mask=batch.attention_mask1,
                                  labels=batch.labels1)
            sent2_outputs = model(input_ids=batch.input_ids2,
                                  token_type_ids=batch.token_type_ids2,
                                  attention_mask=batch.attention_mask2,
                                  labels=batch.labels2)

            train_loss = sent2_outputs.loss - config.r * sent1_outputs.loss

            train_losses.append(accelerator.gather(sent2_outputs.loss.repeat(config.train_batch_size)))

            train_loss = train_loss / config.gradient_accumulation_steps
            accelerator.backward(train_loss)

            if accumulation_step % config.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)

                # Evaluation
                if i_step % config.eval_per_step == 0:
                    model.eval()
                    eval_losses = []
                    for step, batch_eval in enumerate(eval_dataloader):
                        with torch.no_grad():
                            outputs = model(input_ids=batch_eval['input_ids2'],
                                            token_type_ids=batch_eval['token_type_ids2'],
                                            attention_mask=batch_eval['attention_mask2'],
                                            labels=batch_eval['labels2'])

                        eval_loss = outputs.loss
                        eval_losses.append(accelerator.gather(eval_loss.repeat(config.eval_batch_size)))

                    eval_losses = torch.cat(eval_losses)
                    eval_losses = eval_losses[: len(config.eval_dataloader)]
                    try:
                        eval_perplexity = math.exp(torch.mean(eval_losses))
                    except OverflowError:
                        eval_perplexity = float("inf")

                    train_losses = torch.cat(train_losses)
                    train_losses = train_losses[: config.eval_per_step]
                    try:
                        train_perplexity = math.exp(torch.mean(train_losses))
                    except OverflowError:
                        train_perplexity = float("inf")

                    if accelerator.is_main_process:
                        logging.info(
                            f"Step {i_step}: Train Perplexity: {train_perplexity}, Val Perplexity: {eval_perplexity}\n")

                    train_losses = []
                    model.train()

                # Save model
                if i_step % config.save_per_step == 0:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(os.path.join(config.model_save_path, str(i_step)),
                                                    save_function=accelerator.save)

                i_step += 1

            accumulation_step += 1
