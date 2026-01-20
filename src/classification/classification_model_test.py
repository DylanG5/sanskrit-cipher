from classification_POC import predict_image


image_test_list = [
    "./OR15009_112B_L_BLL120.jpg",
    "./OR15010_15V_L_BLL71.jpg",
    "./OR15010_201B_L_BLL100.jpg"
]

model_path = "./output/best_model.pt"
# model_path = "./output/final_model.pt"
meta_path = "./output/meta.json"

for image_path in image_test_list:
    print("\n")
    print("=" * 50)
    print(f"Predicting for image: {image_path}")
    print("=" * 50)

    predicted_lines, confidence, all_probs = predict_image(
        model_path, image_path, meta_path)

    print(f"\nPredicted number of lines: {predicted_lines}")
    print(f"Confidence: {confidence:.2%}")
    print("\nTop 5 confidence scores:")
    print("-" * 40)
    for line_count, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{line_count} lines --> {prob:.2%} confidence")
