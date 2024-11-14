import json

# Charger les données du dataset Piaf
with open("data/piaf.json", "r", encoding="utf-8") as file:
    piaf_data = json.load(file)

# Transformer les données en format question-réponse compatible
formatted_data = []
for article in piaf_data["data"]:
    for paragraph in article["paragraphs"]:
        for qa in paragraph["qas"]:
            question = qa["question"].strip()
            if "answers" in qa and qa["answers"]:
                # Prendre la première réponse disponible
                answer = qa["answers"][0]["text"].strip()
                formatted_data.append({"question": question, "reponse": answer})

# Sauvegarder les données transformées en un fichier JSON compatible
with open("data/data.json", "w", encoding="utf-8") as outfile:
    json.dump(formatted_data, outfile, ensure_ascii=False, indent=4)

print("Transformation terminée. Les données sont maintenant compatibles avec le modèle.")
