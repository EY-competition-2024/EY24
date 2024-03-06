##############      Configuración      ##############
import os

# REPO = r"/mnt/d/Becas y Proyectos/EY Challenge 2024/EY24"
REPO = r"C:/Users/Usuario/OneDrive - Royal Holloway University of London/EY Deep Learning/ey_deep_learning/EY24"
assert os.path.isdir(
    REPO
), "No existe el repositorio. Revisar la variable REPO del codigo run_model"

PATH_DATAIN = rf"{REPO}/data/data_in"
PATH_DATAOUT = rf"{REPO}/data/data_out"
PATH_SCRIPTS = rf"{REPO}/scripts"
PATH_LOGS = rf"{REPO}/logs"
PATH_OUTPUTS = rf"{REPO}/outputs"

for folder in [PATH_DATAIN, PATH_DATAOUT, PATH_SCRIPTS, PATH_LOGS, PATH_OUTPUTS]:
    os.makedirs(folder, exist_ok=True)

###############################################


def visualize_dataset(savename, inputs, value_range, rows, cols):
    import matplotlib.pyplot as plt
    import numpy as np

    inputs = next(iter(inputs.take(1)))
    images, y_true = inputs[0], inputs[1]
    images = images.numpy()
    bboxs = y_true["boxes"].numpy()
    im_classes = y_true["classes"].numpy()

    fig, ax = plt.subplots(rows, cols, figsize=(rows * 5, cols * 5))
    for i, ax in enumerate(ax.flatten()):
        boxes = bboxs[i]
        classes = im_classes[i]

        for j, bbox in enumerate(boxes):

            damaged = classes[j]
            if damaged:
                edgecolor = "r"
            else:
                edgecolor = "b"

            box = plt.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                fill=None,
                linewidth=2,
                edgecolor=edgecolor,
            )
            ax.add_patch(box)

        ax.imshow(images[i])  # Specify extent here
        ax.set_axis_off()

    out = f"{PATH_OUTPUTS}/{savename}_example_imgs"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Se creo {out}")


def visualize_predictions(savename, images, y_preds, rows, cols):
    import earthpy.plot as ep
    import matplotlib.pyplot as plt
    import numpy as np

    images = images
    bboxs = y_preds["boxes"]
    im_classes = y_preds["classes"]
    confidences = y_preds["confidence"]

    fig, ax = plt.subplots(
        rows, cols, figsize=(5 * cols, 5 * rows), constrained_layout=True
    )
    for i, ax in enumerate(ax.flatten()):
        boxes = bboxs[i]
        classes = im_classes[i]
        confidence = confidences[i]
        for j, bbox in enumerate(boxes):

            im_class = classes[j]
            if im_class == -1:
                continue
            if im_class == 1:
                edgecolor = "r"
            else:
                edgecolor = "b"

            box = plt.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                fill=None,
                linewidth=2,
                edgecolor=edgecolor,
            )
            ax.add_patch(box)

            ax.text(
                bbox[0] + 3,
                bbox[3] - 3,
                round(confidence[j], 2),
                fontsize=10,
                color=edgecolor,
            )

        ax.imshow(images[i])  # Specify extent here
        ax.set_axis_off()

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    out = f"{PATH_OUTPUTS}/{savename}_submitions"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Se creo {out}")


def visualize_merged_predictions(savename, rows, cols):
    import earthpy.plot as ep
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    import cv2

    # Load images
    submit_folder_in = rf"{PATH_DATAIN}\Submission data"
    images_files = os.listdir(submit_folder_in)
    images = []
    for image_file in images_files:
        image_path = os.path.join(submit_folder_in, image_file)
        image = cv2.imread(image_path)
        images += [image]
    images = np.stack(images)

    # Load txt files
    txt_files_content = {}
    submit_folder_out = rf"{PATH_DATAOUT}\Submission data\{savename}_formatted_submission"
    text_filenames = os.listdir(submit_folder_out)
    text_filenames = [filename for filename in text_filenames if filename.endswith('.txt')]
    for text_filename in text_filenames:
        if text_filename.endswith(".txt"):
            file_path = os.path.join(submit_folder_out, text_filename)
            with open(file_path, 'r') as file:
                    # Split the file into lines and then split each line into a list of words
                    txt_files_content[text_filename] = [line.strip().split() for line in file.readlines()]

    class_color_encoding = {'undamagedresidentialbuilding': 'orange', 'damagedresidentialbuilding': 'red',
                            'undamagedcommercialbuilding': 'aquamarine', 'damagedcommercialbuilding': 'purple'}

    fig, ax = plt.subplots(
        rows, cols, figsize=(5 * cols, 5 * rows), constrained_layout=True
    )

    # Create patches for the legend
    legend_patches = [patches.Patch(color=color, label=class_name) for class_name, color in class_color_encoding.items()]

    # Loop through images and predictions
    for i, ax in enumerate(ax.flatten()):

        merged_predictions = txt_files_content[text_filenames[i]]

        # Loop through predicted bounding boxes
        for prediction in merged_predictions:

            im_class = prediction[0]
            edgecolor = class_color_encoding[im_class]

            confidence = float(prediction[1])

            bbox = [float(n) for n in prediction[2:6]]

            box = plt.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                fill=None,
                linewidth=2,
                edgecolor=edgecolor,
            )
            
            ax.add_patch(box)

            ax.text(
                bbox[0] + 3,
                bbox[3] - 3,
                round(confidence, 2),
                fontsize=10,
                color=edgecolor,
            )

        # ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB), origin='lower')
        # ax.set_axis_off()

    fig.legend(handles=legend_patches, loc='lower center', ncol=len(class_color_encoding), bbox_to_anchor=(0.5, -0.05))
    # plt.show()

    out = f"{PATH_DATAOUT}\Submission data\{savename}_merged_prediction.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Se creo {out}")
