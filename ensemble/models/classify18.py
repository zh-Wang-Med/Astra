import os
import json
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import SwinModel
# from lightning_tools.optim import config_optimizer
import torch.nn.functional as F
from models.labeler18 import labeler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
import csv
import numpy as np
import ipdb
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
    confusion_matrix
)
from sklearn.metrics import (
roc_auc_score,
roc_curve,
precision_recall_curve,
multilabel_confusion_matrix,
classification_report,
)
import math
from lavis.models.blip_models.vit import ViT
from transformers import get_cosine_schedule_with_warmup


class Image_classify(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.labeler = None
        if args.vision_encoder == "merlin":
            self.labeler = labeler(self.args, 0.3)
        if args.vision_encoder == "ct_clip":
            self.labeler = labeler(self.args, 0.3)
        if args.vision_encoder == "fvlm":
            self.labeler = ViT(
                in_channels=1,
                img_size=(112, 256, 352),
                patch_size=(16, 16, 32),
                num_classes=18,
                dropout_rate=0.1,
                qkv_bias=True,
                classification= True,
                post_activation="ignore",
            )
            ckpt = torch.load('../huggingface/repos/fVLM/model.pth',map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))
            new_ckpt = {}
            for key, value in ckpt['model'].items():
                if key.startswith('visual_encoder'):
                    new_key = key.replace('visual_encoder.', '', 1)
                    new_ckpt[new_key] = value
            self.labeler.load_state_dict(new_ckpt, strict=False)

        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0
        self.predict_study_id_list = []
        self.predict_report_list = []
        if self.args.dataset == 'ctrate':
            self.label_names = [
                'Medical_material', 'Arterial_wall_calcification', 'Cardiomegaly', 'Pericardial_effusion',
                'Coronary_artery_wall_calcification', 'Hiatal_hernia', 'Lymphadenopathy', 'Emphysema',
                'Atelectasis', 'Lung_nodule', 'Lung_opacity', 'Pulmonary_fibrotic_sequela',
                'Pleural_effusion', 'Mosaic_attenuation_pattern', 'Peribronchial_thickening',
                'Consolidation', 'Bronchiectasis', 'Interlobular_septal_thickening'
            ]
            self.disease_num = 18
        if self.args.dataset == 'radchest':
            self.label_names = [
                'Medical_material', 'Cardiomegaly', 'Pericardial_effusion',
                'Hiatal_hernia', 'Lymphadenopathy', 'Emphysema',
                'Atelectasis', 'Lung_nodule', 'Lung_opacity', 'Pulmonary_fibrotic_sequela',
                'Pleural_effusion', 'Peribronchial_thickening',
                'Consolidation', 'Bronchiectasis', 'Interlobular_septal_thickening', 'calcification'
            ]
            self.disease_num = 16
        if self.args.dataset == 'merlin':
            self.label_names = [
                'submucosal_edema', 'renal_hypodensities', 'aortic_valve_calcification',
                'coronary_calcification', 'thrombosis', 'metastatic_disease',
                'pancreatic_atrophy', 'renal_cyst', 'osteopenia',
                'surgically_absent_gallbladder', 'atelectasis', 'abdominal_aortic_aneurysm',
                'anasarca', 'hiatal_hernia', 'lymphadenopathy',
                'prostatomegaly', 'biliary_ductal_dilation', 'cardiomegaly',
                'splenomegaly', 'hepatomegaly', 'atherosclerosis',
                'ascites', 'pleural_effusion', 'hepatic_steatosis',
                'appendicitis', 'gallstones', 'hydronephrosis',
                'bowel_obstruction', 'free_air', 'fracture'
            ]
            self.disease_num = 30
        if self.args.dataset == 'rsna_pe':
            self.label_names = ['leftsided_pe','rightsided_pe','central_pe','chronic_pe']
            self.disease_num = 4
        self.predict_lists = {label: [] for label in self.label_names}
        self.predictroad = args.predictroad
        if self.args.dataset == 'ctrate':
            self.pos_weight = torch.tensor([9.211362733, 2.384068466, 8.295479204, 32.8629776, 2.992233613,  # ct rate版本
                                6.064870808, 3.176470588, 4.187083754, 3.022222222, 1.216071737,
                                1.677849552, 3.152851834, 7.123261694, 18.16629381, 13.8480647,
                                6.335045662, 10.81701149, 13.40695067]).cuda()
        if self.args.dataset == 'radchest':
            self.pos_weight = torch.tensor([2.140, 7.562, 5.178, 7.467, 4.802, 2.675, 2.308, 0.271, 0.868,
            6.166, 3.980, 9.732, 6.211, 5.229, 10, 0.436]).cuda()
        if self.args.dataset == 'merlin':
            self.pos_weight = torch.tensor([
                30.000, 30.000, 30.000, 30.000, 30.000, 30.000, 30.000, 8.211, 30.000, 5.032,
                4.684, 30.000, 13.900, 16.077, 30.000, 30.000, 30.000, 30.000, 30.000, 30.000,
                5.083, 13.513, 5.110, 18.209, 30.000, 30.000, 30.000, 30.000, 30.000, 30.000
            ]).cuda()
        if self.args.dataset == 'rsna_pe':
            self.pos_weight = torch.tensor([3.6024, 2.8277, 16.4835, 23.3598]).cuda()

        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))[
                'model']
            self.load_state_dict(state_dict=state_dict, strict=True)
            print(f'Load checkpoint from {args.delta_file}')

    def CE_loss(self, x, y):
        loss = nn.BCEWithLogitsLoss(pos_weight = self.pos_weight.to(x.device))
        return loss(x,y)

    def forward(self, samples):
        image = samples["image"]
        # ipdb.set_trace()
        labels = {name: samples[name] for name in self.label_names}

        if self.args.text_help:
            text_embedding = samples["text_embedding"]
            predictions = self.labeler.forward_with_text(image,text_embedding)
        else:
            predictions = self.labeler(image)
        label_list = []
        for name in self.label_names:
            label = samples[name]  
            label = label.view(-1, 1)  
            label_list.append(label)
    
        labels = torch.cat(label_list, dim=1).float()  # (B, 18)
        total_loss = self.CE_loss(predictions,labels)
        # ipdb.set_trace()

        return {"loss": total_loss}

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=True)
        return result

    def save_checkpoint(self, eval_res):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        state_dict = self.state_dict()
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step": global_step
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'pths'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'pths',
            "checkpoint_epoch{}_step{}_f1{:3f}.pth".format(current_epoch, global_step, eval_res),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)

    def validation_step(self, samples, batch_idx):
        image = samples["image"]

        if self.args.text_help:
            text_embedding = samples["text_embedding"]
            predictions = self.labeler.forward_with_text(image, text_embedding)
        else:
            predictions = self.labeler(image)

        processed_data = {}

        pred_logits_per_class = torch.unbind(predictions, dim=1)

        for idx, name in enumerate(self.label_names):
            pred_logit = pred_logits_per_class[idx]
            pred_prob = torch.sigmoid(pred_logit)
            pred_label = (pred_prob > self.args.threshold).float()

            true_label = samples[name].view(-1).float()
            processed_data[name] = (true_label.cpu(), pred_label.cpu(), pred_prob.cpu())

        self.val_step_outputs.append(processed_data)

        return processed_data

    def on_validation_epoch_end(self):
        all_labels = {name: [] for name in self.label_names}
        all_preds = {name: [] for name in self.label_names}
        all_probs = {name: [] for name in self.label_names} 

        for batch_output in self.val_step_outputs:
            for name in self.label_names:
                # true_labels, pred_labels = batch_output[name]
                # all_labels[name].append(true_labels)
                # all_preds[name].append(pred_labels)
                true_labels, pred_labels, pred_probs = batch_output[name] 
                all_labels[name].append(true_labels)
                all_preds[name].append(pred_labels)
                all_probs[name].append(pred_probs)


        all_labels_list = []
        all_preds_list = []
        all_probs_list = []

        for name in self.label_names:
            labels = torch.cat(all_labels[name])
            preds = torch.cat(all_preds[name])
            
            
            all_labels_list.append(labels)
            all_preds_list.append(preds)
            all_probs_list.append(torch.cat(all_probs[name]))

        all_labels_tensor = torch.stack(all_labels_list, dim=1)
        all_preds_tensor = torch.stack(all_preds_list, dim=1)
        all_probs_tensor = torch.stack(all_probs_list, dim=1)

        all_labels_array = all_labels_tensor.cpu().numpy()
        all_preds_array = all_preds_tensor.cpu().numpy()
        all_probs_array = all_probs_tensor.numpy() 


        try:
            if self.disease_num == 1:
                val_auc = roc_auc_score(all_labels_array.flatten(), all_probs_array.flatten())
                print(f"Validation AUC: {val_auc:.4f}")
            else:
                val_auc = roc_auc_score(all_labels_array, all_probs_array, average='macro')
                print(f"Validation Macro AUC: {val_auc:.4f}")
        except ValueError as e:
            val_auc = 0.0
            print(f"AUC calculation error (possibly single class in data): {e}")

        cm = multilabel_confusion_matrix(all_labels_array, all_preds_array)
        if self.disease_num != 1:
            clf = classification_report(all_labels_array, all_preds_array, target_names=self.label_names)
            print(clf)

        precision_list = []
        recall_list = []
        f1_list = []
        support_list = []
        for matrix in cm:
            TN, FP, FN, TP = matrix.ravel()
            precision = TP / (TP + FP) if (TP + FP) != 0 else 0
            recall = TP / (TP + FN) if (TP + FN) != 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
            support = TP + FN

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            support_list.append(support)

        total_support = np.sum(support_list)
        weighted_precision = np.sum([precision * support for precision, support in zip(precision_list, support_list)]) / total_support
        weighted_recall = np.sum([recall * support for recall, support in zip(recall_list, support_list)]) / total_support
        weighted_f1 = np.sum([f1 * support for f1, support in zip(f1_list, support_list)]) / total_support
        print(weighted_precision)
        print(weighted_recall)
        print(weighted_f1)
        self.val_step_outputs.clear()

        if self.trainer.local_rank == 0:
            self.save_checkpoint(weighted_f1)
            self.val_score = weighted_f1
        self.val_step_outputs.clear()


    def test_step(self, samples, batch_idx):
        image = samples["image"]

        if self.args.text_help:
            text_embedding = samples["text_embedding"]
            predictions = self.labeler.forward_with_text(image,text_embedding)
        else:
            predictions= self.labeler(image)


        processed_data = {}

        pred_logits_per_class = torch.unbind(predictions, dim=1)  

        for idx, name in enumerate(self.label_names):
            pred_logit = pred_logits_per_class[idx]  
            pred_prob = torch.sigmoid(pred_logit)   
            pred_label = pred_prob  # (batch_size,)

            true_label = samples[name].view(-1)  # (batch_size,)

            true_label = true_label.cpu()
            pred_label = pred_label.cpu()
            processed_data[name] = (true_label, pred_label)

        self.test_step_outputs.append(processed_data)

        return processed_data


    def on_test_epoch_end(self):
        all_labels = {name: [] for name in self.label_names}
        all_scores = {name: [] for name in self.label_names}

        for batch_output in getattr(self, "test_step_outputs", []):
            for name in self.label_names:
                true_labels, pred_scores = batch_output[name]

                if isinstance(true_labels, np.ndarray):
                    true_labels = torch.from_numpy(true_labels)
                if isinstance(pred_scores, np.ndarray):
                    pred_scores = torch.from_numpy(pred_scores)

                true_labels = true_labels.float().view(-1)
                pred_scores = pred_scores.float().view(-1)

                all_labels[name].append(true_labels)
                all_scores[name].append(pred_scores)

        all_labels_list = []
        all_scores_list = []
        for name in self.label_names:
            labels = torch.cat(all_labels[name], dim=0)
            scores = torch.cat(all_scores[name], dim=0)

            all_labels_list.append(labels)
            all_scores_list.append(scores)

        y_true = torch.stack(all_labels_list, dim=1).cpu().numpy().astype(int)   # N x C
        y_score = torch.stack(all_scores_list, dim=1).cpu().numpy()              # N x C

        thresholds = []
        per_class_auc = []
        per_class_opt = [] 

        for i, name in enumerate(self.label_names):
            yt = y_true[:, i]
            ys = y_score[:, i]

            if len(np.unique(yt)) < 2:
                thresholds.append(0.5)
                per_class_auc.append(np.nan)
                per_class_opt.append({"metric": "F1", "value": np.nan})
                continue

            method_used = "F1"
            try:
                precision, recall, pr_thresholds = precision_recall_curve(yt, ys)
                f1_scores = np.where(
                    (precision + recall) > 0, 2 * precision * recall / (precision + recall), 0
                )
                f1_for_thr = f1_scores[:-1]
                if f1_for_thr.size > 0:
                    best_idx = np.argmax(f1_for_thr)
                    best_thr = pr_thresholds[best_idx]
                    best_metric_value = f1_for_thr[best_idx]
                else:
                    raise ValueError("No thresholds from PR curve")
            except Exception:
                fpr, tpr, roc_thresholds = roc_curve(yt, ys)
                j = tpr - fpr
                best_idx = np.argmax(j)
                best_thr = roc_thresholds[best_idx]
                best_metric_value = j[best_idx]
                method_used = "YoudenJ"

            try:
                auc_i = roc_auc_score(yt, ys)
            except Exception:
                auc_i = np.nan

            thresholds.append(float(best_thr))
            per_class_auc.append(float(auc_i))
            per_class_opt.append({"metric": method_used, "value": float(best_metric_value)})

        try:
            micro_auc = float(roc_auc_score(y_true, y_score, average="micro"))
        except Exception:
            micro_auc = np.nan

        try:
            macro_auc = float(roc_auc_score(y_true, y_score, average="macro"))
        except Exception:
            valid_aucs = []
            for i in range(y_true.shape[1]):
                yt = y_true[:, i]
                ys = y_score[:, i]
                if len(np.unique(yt)) < 2:
                    continue
                try:
                    valid_aucs.append(roc_auc_score(yt, ys))
                except Exception:
                    pass
            macro_auc = float(np.mean(valid_aucs)) if valid_aucs else np.nan

        thr_array = np.array(thresholds, dtype=np.float32)
        y_pred_bin = (y_score >= thr_array).astype(int)

        os.makedirs(self.args.savedmodel_path, exist_ok=True)
        np.savez(
            os.path.join(self.args.savedmodel_path, "classification_results.npz"),
            labels=y_true,
            probs=y_score,
            thresholds=thr_array,
            binary_preds=y_pred_bin,
        )

        cm = multilabel_confusion_matrix(y_true, y_pred_bin)
        if self.disease_num != 1:
            clf = classification_report(
                y_true, y_pred_bin, target_names=self.label_names, zero_division=0
            )

        report_lines = []
        report_lines.append("Per-class best thresholds and AUC:")
        for i, name in enumerate(self.label_names):
            report_lines.append(
                f"{name}: threshold={thresholds[i]:.6f}, AUC={per_class_auc[i]}, "
                f"opt_metric={per_class_opt[i]['metric']}, opt_value={per_class_opt[i]['value']}"
            )
        report_lines.append(f"\nGlobal micro AUC: {micro_auc}")
        report_lines.append(f"Global macro AUC: {macro_auc}\n")
        report_lines.append("Classification report with per-class best thresholds:\n")
        if self.disease_num != 1:
            report_lines.append(clf)

        with open(os.path.join(self.args.savedmodel_path, "test_classification_report.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        print(f"Global micro AUC: {micro_auc}")
        print(f"Global macro AUC: {macro_auc}")
        if self.disease_num != 1:
            print(clf)

        self.test_step_outputs = []



    def predict_step(self, samples, batch_idx):
        study_id = samples["id"]          # (batch_size,)
        image = samples["image"]          # (batch_size, C, H, W)

        predictions = self.labeler(image)  # (B, 18)

        pred_probs = torch.sigmoid(predictions)  # (B, 18)

        pred_labels = (pred_probs > self.args.threshold).float()  # (B, 18)

        pred_per_class = torch.unbind(pred_labels, dim=1)  # 18 × (B,)

        self.predict_study_id_list.extend(study_id)  # 假设 study_id 是 list 或 tuple

        for idx, name in enumerate(self.label_names):
            preds = pred_per_class[idx].cpu().tolist()  # list of float or int (0/1)
            self.predict_lists[name].extend(preds)



    def on_predict_epoch_end(self):
        all_results = []

        header = ["study_id"] + self.label_names
        all_results.append(header)

        print(len(self.predict_study_id_list))
        for i, study_id in enumerate(self.predict_study_id_list):
            row = [study_id]
            for name in self.label_names:
                row.append(self.predict_lists[name][i])
            all_results.append(row)

        import csv
        import os

        os.makedirs(self.args.savedmodel_path, exist_ok=True)

        output_file = os.path.join(self.args.savedmodel_path, "val_prediction_results.csv")

        with open(output_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerows(all_results)

        print(f"预测结果已保存到: {output_file}")

        self.predict_study_id_list.clear()
        for name in self.label_names:
            self.predict_lists[name].clear()



    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        
        total_steps = self.trainer.estimated_stepping_batches
        
        warmup_steps = int(total_steps * 0.05) 
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()


