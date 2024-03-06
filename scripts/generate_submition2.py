if __name__ == "__main__":

    # from run_model import generate_predictions
    from merge_models_predictions import generate_txt_for_submission
    from plot_tools import visualize_merged_predictions

    savename_damaged = "YOLOv8_sample20000_damaged_fine_tuning"
    # generate_predictions(savename_damaged)

    savename_commercial = "YOLOv8_sample20000_commercial_fine_tuning"
    # generate_predictions(savename_commercial)

    generate_txt_for_submission(savename_model1 = savename_damaged, savename_model2 = savename_commercial,
                                conserve_predictions_without_match=False, pct_overlap_threshold=0.1) # PARAMETROS IMPORTANTES!!!

    # Separate the string in savename_damaged by "_"
    parts = savename_damaged.split("_")
    # Delete the part that contains "damaged" or "commercial"
    filtered_parts = [part for part in parts if "damaged" not in part and "commercial" not in part]
    # Re-join the parts with "_"
    savename = "_".join(filtered_parts)

    visualize_merged_predictions(savename, 4, 3)