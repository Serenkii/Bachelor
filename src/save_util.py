from typing import overload

from pypdf import PdfReader, PdfWriter

import matplotlib.pyplot as plt
import matplotlib as mpl


def get_metadata(file_path, keys=None):
    reader = PdfReader(file_path)

    info = reader.metadata

    return_dict = dict()
    if keys:
        for key in keys:
            if key not in info.keys():
                continue
            return_dict[key] = info[key]
    else:
        return_dict = info

    return return_dict


def edit_metadata(file_path, metadata):

    reader = PdfReader(file_path)
    writer = PdfWriter()

    for page in reader.pages:
        writer.add_page(page)

    info = reader.metadata

    new_metadata = info or dict()
    new_metadata.update(metadata)

    writer.add_metadata(new_metadata)

    with open(file_path, "wb") as f:
        writer.write(f)

    return new_metadata


def savefig(fig, file_path, source_paths, description, **kwargs):
    fig.savefig(file_path, **kwargs)

    metadata = { "/SourcePaths": source_paths, "/Description": description }

    edit_metadata(file_path, metadata)


def _getset(file_path, key, update=None):
    if update:
        return edit_metadata(file_path, {key: update})
    metadata = get_metadata(file_path, [key])
    print(metadata)
    return metadata


def source_paths(file_path, update=None):
    key = "/SourcePaths"
    return _getset(file_path, key, update)

def description(file_path, update=None):
    key = "/Description"
    return _getset(file_path, key, update)









# %%
test = True

if __name__ == '__main__' and test:
    print(get_metadata("/home/user/marian.gunsch/PycharmProjects/Bachelor/out/thesis/equilibrium/comparison_B_field.pdf"))

    edit_metadata("/home/user/marian.gunsch/PycharmProjects/Bachelor/out/thesis/equilibrium/comparison_B_field.pdf",
                  {"/SourcePath": "test"})

    print(get_metadata("/home/user/marian.gunsch/PycharmProjects/Bachelor/out/thesis/equilibrium/comparison_B_field.pdf"))

