import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import argparse

slice_num = 70


def imprimir_inferencia(serie, seg_out, seg, slice, recurrence=False):
    global slice_num
    if recurrence:
        s = 2
        path = os.path.join(
            "./Dataset",
            f"Dataset_106_10_casos/test/images/images_structural/UPENN-GBM-{serie}_11/UPENN-GBM-{serie}_11_T1GD.nii.gz",
        )
        path_recurrence = os.path.join(
            "./Dataset",
            f"Dataset_106_10_casos/recurrence/images_structural/UPENN-GBM-{serie}_21/UPENN-GBM-{serie}_21_T1GD.nii.gz",
        )
    else:
        s = 1
        path = os.path.join(
            "./Dataset",
            f"Dataset_106_10_casos/test/images/images_structural/UPENN-GBM-{serie}_{s}1/UPENN-GBM-{serie}_{s}1_T1GD.nii.gz",
        )

    img_add = path
    if recurrence:
        img_rec = path_recurrence

    label_add = os.path.join(
        "./Dataset",
        f"Dataset_106_10_casos/test/labels/UPENN-GBM-{serie}_11_segm.nii.gz",
    )
    if not os.path.exists(label_add):
        label_add = os.path.join(
            "./Dataset",
            f"Dataset_106_10_casos/test/labels/UPENN-GBM-{serie}_11_automated_approx_segm.nii.gz",
        )

    img = nib.load(img_add).get_fdata()
    if recurrence:
        img_rec = nib.load(img_rec).get_fdata()

    label = nib.load(label_add).get_fdata()

    fig, ax = plt.subplots(2, 3, figsize=(18, 18))
    fig.canvas.mpl_connect(
        "scroll_event", lambda event: scroll_slices(event, img.shape[2])
    )
    fig.canvas.mpl_connect(
        "key_press_event", lambda event: key_press(event, img.shape[2])
    )

    def update_slice(slice_num):
        ax[0, 0].clear()
        ax[0, 0].imshow(img[:, :, slice_num], cmap="gray")
        ax[0, 0].set_title("image")

        ax[0, 1].clear()
        ax[0, 1].imshow(label[:, :, slice_num])
        ax[0, 1].set_title("label")

        ax[0, 2].clear()
        ax[0, 2].imshow(seg_out[:, :, slice_num])
        ax[0, 2].set_title("nroi - froi - inter.")

        ax[1, 0].clear()
        ax[1, 0].imshow(seg[0][:, :, slice_num])
        ax[1, 0].set_title("Map nroi")

        ax[1, 1].clear()
        ax[1, 1].imshow(seg[1][:, :, slice_num])
        ax[1, 1].set_title("Map froi")

        if recurrence:
            ax[1, 2].clear()
            ax[1, 2].imshow(img_rec[:, :, slice_num], cmap="gray")
            ax[1, 2].set_title("image_recurrence")

        plt.draw()

    def scroll_slices(event, max_slices):
        global slice_num
        if event.button == "up":
            slice_num = (slice_num + 1) % max_slices
            update_slice(slice_num)
        elif event.button == "down":
            slice_num = (slice_num - 1) % max_slices
            update_slice(slice_num)

    def key_press(event, max_slices):
        global slice_num
        if event.key == "up":
            slice_num = (slice_num + 1) % max_slices
            update_slice(slice_num)
        elif event.key == "down":
            slice_num = (slice_num - 1) % max_slices
            update_slice(slice_num)

    update_slice(slice_num)
    plt.show()


def main():
    global slice_num
    parser = argparse.ArgumentParser(description="Visualizador de MRI")
    parser.add_argument("--serie", type=int, default=36, help="Número de serie")
    args = parser.parse_args()

    seg_out = np.load(f"trained_models/seg_out_{str(args.serie).zfill(5)}.npy")
    seg = np.load(f"trained_models/seg_{str(args.serie).zfill(5)}.npy")
    slice = 70  # Slice inicial

    imprimir_inferencia(
        f"{str(args.serie).zfill(5)}", seg_out, seg, slice, recurrence=True
    )


if __name__ == "__main__":
    main()
