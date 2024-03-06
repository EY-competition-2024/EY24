##############      Configuración      ##############
import os

# REPO = r"/mnt/d/Becas y Proyectos/EY Challenge 2024/EY24"
REPO = rf"C:\Users\Usuario\OneDrive - Royal Holloway University of London\EY Deep Learning\ey_deep_learning\EY24"
assert os.path.isdir(
    REPO
), "No existe el repositorio. Revisar la variable REPO del codigo match_models_predictions"

PATH_DATAIN = rf"{REPO}/data/data_in"
PATH_DATAOUT = rf"{REPO}/data/data_out"
PATH_SCRIPTS = rf"{REPO}/scripts"
PATH_LOGS = rf"{REPO}/logs"
PATH_OUTPUTS = rf"{REPO}/outputs"

for folder in [PATH_DATAIN, PATH_DATAOUT, PATH_SCRIPTS, PATH_LOGS, PATH_OUTPUTS]:
    os.makedirs(folder, exist_ok=True)

###############################################
    
import os
import numpy as np
    
def generate_txt_for_submission(savename_model1, savename_model2,
                                conserve_predictions_without_match=False, pct_overlap_threshold = 0.01):
    
    # Create folder for submissions
    submit_folder_out = rf"{PATH_DATAOUT}/Submission data"
    if not os.path.exists(submit_folder_out):
        os.makedirs(submit_folder_out)

    # Create the name of the folder final submission for this particular model
        
    # Separate the string by "_"
    parts = savename_model1.split("_")
    # Delete the part that contains "damaged" or "commercial"
    filtered_parts = [part for part in parts if "damaged" not in part and "commercial" not in part]
    # Re-join the parts with "_"
    savename_submission = "_".join(filtered_parts)
    # Create folder
    formated_submission_folder = os.path.join(submit_folder_out, savename_submission + '_formatted_submission')
    if not os.path.exists(formated_submission_folder):
        os.makedirs(formated_submission_folder)

    # Get filenames for submission images
    submit_folder_in = rf"{PATH_DATAIN}/Submission data"
    print(submit_folder_in)
    images_filenames = os.listdir(submit_folder_in)

    # Get the predictions by both models
    pred_model1 = np.load(rf"{PATH_DATAOUT}\{savename_model1}_unformatted_submitions.npy", allow_pickle=True).item()
    pred_model2 = np.load(rf"{PATH_DATAOUT}\{savename_model2}_unformatted_submitions.npy", allow_pickle=True).item()
    
    # For each image, created the matched prediction and save it as a formatted txt file
    for i, image_filename in enumerate(images_filenames):

        # Find the matching bounding boxes between the predictions of model 1 and model 2 (with model 1 as benchmark)
        matched_boxes = match_intersecting_bboxes(pred_model1['boxes'][i], pred_model2['boxes'][i], 
                                                  conserve_predictions_without_match=conserve_predictions_without_match,
                                                  pct_overlap_threshold=pct_overlap_threshold)

        # For those matches, construct lists for classes, confidence scores and bounding boxes
        class_names = build_matched_classes(pred_model1['classes'][i], pred_model2['classes'][i], matched_boxes)
        confidence_scores = pred_model1['confidence'][i][list(matched_boxes.keys())]
        bboxes = pred_model1['boxes'][i][list(matched_boxes.keys())]

        assert len(class_names) == len(confidence_scores) == len(bboxes)

        # Creating a new .txt file for each image in the submission_directory
        with open(os.path.join(formated_submission_folder, os.path.splitext(image_filename)[0]) + '.txt', "w") as file:
            for i in range(len(class_names)):
                # Get coordinates of each bounding box
                # left, top, right, bottom = bboxes[i]
                left, bottom, right, top = bboxes[i]
                # Write content to file in desired format
                file.write(f"{class_names[i]} {confidence_scores[i]} {left} {bottom} {right} {top}\n")

def get_bbox_intersection(bb1, bb2):
    """
    Get the coordinates for the Intersection of two bounding boxes.

    Parameters
    ----------
    bb1/bb2 : list
        Values: ['x1', 'x2', 'y1', 'y2']
        The (x1, y1) position is at the bottom left corner,
        the (x2, y2) position is at the top right corner

    Returns
    -------
    list or None
    """

    # Get bottom left and top right coordinates from both boxes
    bb1 = {'x1': bb1[0], 'y1': bb1[1], 'x2': bb1[2], 'y2': bb1[3]}
    bb2 = {'x1': bb2[0], 'y1': bb2[1], 'x2': bb2[2], 'y2': bb2[3]}

    # Check if they are correct
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # Determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_bottom = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_top = min(bb1['y2'], bb2['y2'])

    # If there is no intersection, return None. Else return the intersection
    if x_right < x_left or y_top < y_bottom:
        return None
    else:
        bbox_intersection = [x_left, y_bottom, x_right, y_top]
        return bbox_intersection


def get_bbox_area(bb):
    """
    Calculate the area of a bounding box.

    Parameters
    ----------
    bb1 : list
        Values: ['x1', 'x2', 'y1', 'y2']
        The (x1, y1) position is at the bottom left corner,
        the (x2, y2) position is at the top right corner

    Returns
    -------
    float
    """

    # Get bottom left and top right coordinates from both boxes
    x1, y1, x2, y2 = bb

    # Check if they are correct
    assert x1 < x2
    assert y1 < y2

    return (x2 - x1) * (y2 - y1)


# Define the updated function to perform the required operation
def find_intersecting_boxes(bb_array1, bb_array2):
    """
    Takes two arrays of bounding boxes and identifies intersections.
    
    Parameters:
    - bb_array1: Array of bounding boxes. Each bounding box is a list of [x1, y1, x2, y2].
    - bb_array2: Array of bounding boxes. Each bounding box is formatted similarly to bb_array1.
    
    Returns:
    - A dictionar like this one:
    {2: {'area': 3081.7834, 'intersections': {10: 2758.0115, 25: 191.04353}}},
    where:
        . 2 stands for the index of a bounding box in bb_array1,
        . 'area' is the area of the bounding box,
        . 'intersections' contains a dictionary with indexes of the bboxes in bb_array2 that intersect with the bounding box,
            and the area of the intersection with each one.
    So the full translation would be "bbox 2 in bb_array1 has an area of 3081.7834 and intersects with bbox 10 of bb_array2,
        with an area of intersection of 2758.0115, and bbox 25 of bb_array2, with an area of intersection of 191.04353
    """
    intersections_dict = {}

    for i, bb1 in enumerate(bb_array1):

        if bb1[0] == -1:  # Skip placeholder or unused entries
            continue

        # Calculate bbox area
        bb1_area = get_bbox_area(bb1)

        # Placehorlder to store indexes of intersection bboxes and intersection areas
        # intersecting_results = []
        intersecting_results = {'index':[], 'area_overlap':[]}

        for j, bb2 in enumerate(bb_array2):
            if bb2[0] == -1:  # Skip placeholder or unused entries
                continue

            # Get intersection
            bbox_intersection = get_bbox_intersection(bb1, bb2)

            # Add the index and the area of the intersection over the area of the original bbox
            if bbox_intersection is not None:
                # intersection_result = [j, get_bbox_area(bbox_intersection) / bb1_area]
                # intersecting_results.append(intersection_result)

                intersecting_results['index'].append(j)
                intersecting_results['area_overlap'].append(get_bbox_area(bbox_intersection) / bb1_area)

        intersections_dict[i] = intersecting_results

    return intersections_dict


def match_intersecting_bboxes(bb_array1, bb_array2, 
                              pct_overlap_threshold = 0.01, conserve_predictions_without_match=True):

    # Find the intersections and the area overlap
    intersecting_boxes_example = find_intersecting_boxes(bb_array1, bb_array2)    

    matching_results_dict = {}

    for i, values in intersecting_boxes_example.items():

        # Case for no intersection:
        if len(values['area_overlap']) == 0:
            
            if conserve_predictions_without_match:
                matching_results_dict[i] = None

            continue

        # Find intersection with largest overlap
        i_int_with_biggest_overlap = values['area_overlap'].index(max(values['area_overlap']))
        
        # Match it if the overlap is greater than the threshold
        if values['area_overlap'][i_int_with_biggest_overlap] > pct_overlap_threshold:
            matching_results_dict[i] = values['index'][i_int_with_biggest_overlap]

    return matching_results_dict


def build_matched_classes(class_ndarray1, class_ndarray2, matched_boxes, 
                  model_order=['damage_predictor', 'commercial_predictor'], 
                  decoding = {'damage_predictor': {0: 'undamaged', 1: 'damaged'}, 
                              'commercial_predictor': {0: 'residential', 1: 'commercial'}},
                    default_categories = {'damage_predictor': 'undamaged', 
                                          'commercial_predictor': 'commercial'}
                              ):
    
    class_names= []

    for bb1_i, bb2_i in matched_boxes.items():
        
        bb1_class_n = class_ndarray1[bb1_i]
        bb1_class = decoding[model_order[0]][bb1_class_n]

        # If there is no match, attach default category
        if bb2_i is None:            
                bb2_class = default_categories[model_order[1]]
        
        else:            
            bb2_class_n = class_ndarray2[bb2_i]
            bb2_class = decoding[model_order[1]][bb2_class_n]

        class_names.append(bb1_class + bb2_class + 'building')

    return class_names