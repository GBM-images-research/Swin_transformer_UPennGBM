import os
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider  # SliderEvent


def imprimir_inferencia_out(serie, seg_out, seg, slice_num, recurrence=False):
    if recurrence:
        s = 2
        img_path = os.path.join(
            "./Dataset",
            f"Dataset_106_10_casos/recurrence/images_structural/UPENN-GBM-{serie}_21/UPENN-GBM-{serie}_21_T1GD.nii.gz",
        )
    else:
        s = 1
        img_path = os.path.join(
            "./Dataset",
            f"Dataset_106_10_casos/test/images/images_structural/UPENN-GBM-{serie}_{s}1/UPENN-GBM-{serie}_{s}1_T1GD.nii.gz",
        )

    label_path = os.path.join(
        "./Dataset",
        f"Dataset_106_10_casos/test/labels/UPENN-GBM-{serie}_11_segm.nii.gz",
    )
    if not os.path.exists(label_path):
        label_path = os.path.join(
            "./Dataset",
            f"Dataset_106_10_casos/test/labels/UPENN-GBM-{serie}_11_automated_approx_segm.nii.gz",
        )

    img = nib.load(img_path).get_fdata()
    label = nib.load(label_path).get_fdata()

    fig, axarr = plt.subplots(3, 3, figsize=(18, 18))

    def update(val):
        slice_index = int(slider.val)
        axarr[0, 0].imshow(img[:, :, slice_index], cmap="gray")
        axarr[0, 1].imshow(label[:, :, slice_index])
        axarr[0, 2].imshow(seg_out[:, :, slice_index])
        axarr[1, 0].imshow(seg[0][:, :, slice_index])
        axarr[1, 1].imshow(seg[1][:, :, slice_index])
        fig.canvas.draw_idle()

    # Configuración inicial
    slice_index = slice_num
    axarr[0, 0].imshow(img[:, :, slice_index], cmap="gray")
    axarr[0, 1].imshow(label[:, :, slice_index])
    axarr[0, 2].imshow(seg_out[:, :, slice_index])
    axarr[1, 0].imshow(seg[0][:, :, slice_index])
    axarr[1, 1].imshow(seg[1][:, :, slice_index])

    # Añadir slider para cambiar los slices
    ax_slider = plt.axes([0.1, 0.01, 0.65, 0.03])
    slider = Slider(ax_slider, "Slice", 0, img.shape[2] - 1, valinit=slice_index)

    # Conectar la función de actualización al evento del slider
    slider.on_changed(update)

    # Conectar la función de actualización al evento de scroll del ratón
    fig.canvas.mpl_connect(
        "scroll_event", lambda event: update_slider_on_scroll(event, slider)
    )

    plt.show()


# Función para actualizar el slider al hacer scroll
def update_slider_on_scroll(event, slider):
    if event.button == "up":
        slider.set_val(slider.val + 1)
    elif event.button == "down":
        slider.set_val(slider.val - 1)


if __name__ == "__main__":
    imprimir_inferencia_out("your_serie", seg_out, seg, slice_num)
