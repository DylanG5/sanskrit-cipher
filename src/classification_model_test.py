from classification_POC import predict_image

# Predict on a single image
# image_path = "./classification_training_images/OR15010_48V_L [BLL87].jpg"
image_path = "./IOLSAN859B_L [BLL142].jpg"
# image_path = "./OR15003_147V1_L [BLL2].jpg"
# image_path = "./OR15003_150R1_L [BLL2].jpg"
# image_path = "./OR15007_42R1_L [BLL18].jpg"
# image_path = "./OR15009_677B_ L [BLL231].jpg"
image_test_list = [
    "./IOLSAN859B_L [BLL142].jpg",
    "./OR15003_147V1_L [BLL2].jpg",
    "./OR15003_150R1_L [BLL2].jpg",
    "./OR15007_42R1_L [BLL18].jpg",
    "./OR15009_677B_L [BLL231].jpg"
]





model_path = "./output/best_model.pt" 
meta_path = "./output/meta.json"

for image_path in image_test_list:
    print("\n")
    print("=" * 50)
    print(f"Predicting for image: {image_path}")
    print("=" * 50)

    predicted_lines, confidence, all_probs = predict_image(model_path, image_path, meta_path)

    print(f"\nPredicted number of lines: {predicted_lines}")
    print(f"Confidence: {confidence:.2%}")
    print("\nTop 5 confidence scores:")
    print("-" * 40)
    for line_count, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{line_count} lines --> {prob:.2%} confidence")
    

