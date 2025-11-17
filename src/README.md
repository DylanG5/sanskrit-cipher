# Sanskrit Cipher Source Code

The folders and files for this project are as follows:

## Classification Model Training and Testing

### Training the Classification Model

The `classification_POC.py` script trains a CNN model to predict the number of lines in manuscript fragment images.

#### Basic Usage

```bash
python classification_POC.py
```

This will use default parameters:
- Images directory: `./classification_training_images`
- Labels CSV: `./image_line_count.csv`
- Output directory: `./output`
- Image size: 256x256
- Batch size: 16
- Epochs: 50
- Learning rate: 5e-4

#### Custom Arguments

```bash
python classification_POC.py \
    --images_dir ./path/to/images \
    --labels_csv ./path/to/labels.csv \
    --out_dir ./custom_output \
    --img_size 512 \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-3
```

**Available Arguments:**
- `--images_dir`: Directory containing training images
- `--labels_csv`: CSV file with image filenames and line counts
- `--out_dir`: Directory to save model checkpoints and metadata
- `--img_size`: Size to resize images (default: 256)
- `--batch_size`: Batch size for training (default: 16)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 5e-4)

#### Training Output

The script will output:
1. **Console Output:**
   ```
   Using device: cpu (or cuda if GPU available)
   Epoch 1/50: train_loss=2.1234, train_acc=0.2500, val_loss=1.9876, val_acc=0.3125 [BEST]
   Epoch 2/50: train_loss=1.8765, train_acc=0.3750, val_loss=1.7654, val_acc=0.4375 [BEST]
   ...
   ```

2. **Saved Files in Output Directory:**
   - `best_model.pt`: Model with the best validation accuracy
   - `final_model.pt`: Model from the last epoch
   - `meta.json`: Metadata containing `max_lines` and `img_size`

### Testing the Classification Model

The `classification_model_test.py` script tests the trained model on new images.

#### Setup Test Images

Edit the `image_test_list` in `classification_model_test.py` with the image paths you want to test:

```python
image_test_list = [
    "./IOLSAN859B_L [BLL142].jpg",
    "./OR15003_147V1_L [BLL2].jpg",
    "./OR15003_150R1_L [BLL2].jpg",
    "./OR15007_42R1_L [BLL18].jpg",
    "./OR15009_677B_L [BLL231].jpg"
]
```

#### Run Testing

```bash
python classification_model_test.py
```

#### Testing Output

For each test image, the script will display:

```
==================================================
Predicting for image: ./IOLSAN859B_L [BLL142].jpg
==================================================

Predicted number of lines: 3
Confidence: 85.67%

Top 5 confidence scores:
----------------------------------------
3 lines --> 85.67% confidence
2 lines --> 8.45% confidence
4 lines --> 3.21% confidence
1 lines --> 1.89% confidence
5 lines --> 0.78% confidence
```

**Output Explanation:**
- **Predicted number of lines**: The most likely line count
- **Confidence**: Probability of the top prediction
- **Top 5 confidence scores**: Ranked list of the 5 most likely line counts with their probabilities

#### Switching Between Models

You can test with different model checkpoints by commenting/uncommenting in `classification_model_test.py`:

```python
model_path = "./output/best_model.pt"  # Use best validation model
# model_path = "./output/final_model.pt"  # Use final epoch model
```
