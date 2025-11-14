from classification_POC import predict_image

# Predict on a single image
image_path = "./classification_training_images/OR15010_48V_L [BLL87].jpg"
model_path = "./output/best_model.pt"  # or final_model.pt
meta_path = "./output/meta.json"

predicted_lines, confidence = predict_image(model_path, image_path, meta_path)

print(f"Predicted number of lines: {predicted_lines}")
print(f"Confidence: {confidence:.2%}")