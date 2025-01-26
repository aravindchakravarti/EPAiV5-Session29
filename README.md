# EPAiV5-Session29
Session 29 : Advanced loss functions

# Summary
In this assignment we implement a teacher-student model and a distillation model.
## TLDR

| Item                    | Train Accuracy | Test Accuracy | Change in Accuracy |
|-------------------------|----------------|---------------|--------------------|
| Teacher Model           | 91.23%         | 78.91         | NA                 |
| Student Model           | 80.87%         | 74.60%        | NA                 |
| Distilled Student Model | 85.83          | 79.36         | Train = ~5% Test = ~4.75%|
                                               
# Detailed Explanation
## Teacher Model
Teacher model is 729,712 (929k) parameter model trained for CIFAR-10 dataset 100 epochs with early stopping enabled. The model accuracy is as summarized in above table.
Log extract of last 2 epochs is as follows:
```
---------- prev = 0.28933569669723513 current = 0.27017143309116365 ---------
Epoch 29/100
Loss=0.3211618959903717 Batch_id=24 Accuracy=91.05: 100%|██████████| 25/25 [00:18<00:00,  1.35it/s] --> EPOCH: 28, Avg Training Loss: 0.2634, Avg Time Taken = 467.57ms
Test set: Average loss: 0.7115, Accuracy: 7846/10000 (78.46%)


---------- prev = 0.27017143309116365 current = 0.26343166947364804 ---------
Epoch 30/100
Loss=0.3437854051589966 Batch_id=24 Accuracy=91.23: 100%|██████████| 25/25 [00:18<00:00,  1.32it/s] --> EPOCH: 29, Avg Training Loss: 0.2569, Avg Time Taken = 465.19ms
Test set: Average loss: 0.7036, Accuracy: 7891/10000 (78.91%)

---------- prev = 0.26343166947364804 current = 0.2568626445531845 ---------
Model saved at: /content/drive/MyDrive/EPAi_V5/model_heavy_acc_91.pth
Early stopping triggered!


```

## Student Model
Student model is a 230,192 (230k) parameter model trained for CIFAR-10 dataset 100 epochs with early stopping enabled. The model is **roughly 3 times smaller** than 
teacher model. The model accuracy is as summarized in above table. Logs from last 2 epochs is as follows:

```
Epoch 26/100
Loss=0.5759264230728149 Batch_id=24 Accuracy=80.44: 100%|██████████| 25/25 [00:17<00:00,  1.44it/s] --> EPOCH: 25, Avg Training Loss: 0.5586, Avg Time Taken = 360.99ms
Test set: Average loss: 0.7487, Accuracy: 7449/10000 (74.49%)


---------- prev = 0.5649992895126342 current = 0.5586209177970887 ---------
Epoch 27/100
Loss=0.6129828691482544 Batch_id=24 Accuracy=80.87: 100%|██████████| 25/25 [00:18<00:00,  1.37it/s]
 --> EPOCH: 26, Avg Training Loss: 0.5443, Avg Time Taken = 361.21ms
Test set: Average loss: 0.7521, Accuracy: 7460/10000 (74.60%)


---------- prev = 0.5586209177970887 current = 0.5442671251296997 ---------
Model saved at: /content/drive/MyDrive/EPAi_V5/model_small_acc_80.pth
Early stopping triggered!

```

## Distilled Student Model
Now, student model is trained using knowldege distillation. The teacher model is used as the knowledge source. The student model is trained for 100 epochs with early stopping enabled. KL divergence loss is used as the distillation loss along with cross entropy loss. 

```python
    # Compute losses
    loss_soft = soft_loss(student_probs, teacher_probs) * (T ** 2)  # Scale by T^2
    loss_hard = hard_loss(student_logits, target)
    loss = alpha * loss_hard + (1 - alpha) * loss_soft

    epoch_loss += loss.item()
```
By this process model's train and test accuracy is improved by ~5% and ~4.75% respectively.

```
Loss=0.8674250841140747 Batch_id=24 Accuracy=85.67: 100%|██████████| 25/25 [00:19<00:00,  1.27it/s]---------- prev = 0.86076979637146 current = 0.8424052166938781 ---------
Test set: Average loss: 0.7170, Accuracy: 7957/10000 (79.57%)


Loss=0.9037202000617981 Batch_id=24 Accuracy=85.83: 100%|██████████| 25/25 [00:18<00:00,  1.38it/s]---------- prev = 0.8424052166938781 current = 0.8303352212905883 ---------
Test set: Average loss: 0.6974, Accuracy: 7936/10000 (79.36%)
```