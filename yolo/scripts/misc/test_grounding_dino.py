from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology

ontology = CaptionOntology({
    "pallet": "wooden pallet at the bottom without boxes on top",
})

custom_model_path = "./groundingdino_swint_ogc.pth"


# Define the base model and ontology for pallet detection
base_model = GroundingDINO(ontology=ontology, box_threshold=0.25)

# Paths for input images and labeled output
input_folder = "./dataset/Pallets_10"  # Path to your folder containing images
output_folder = "./autodistill_outputs_pallets"  # Path to save labeled outputs

# Generate labels for the images
base_model.label(input_folder=input_folder, output_folder=output_folder, extension=".jpg")

# 
print(f"Pallet labels generated and saved in: {output_folder}")
