import numpy as np
import thesis.mpl_configuration

import matplotlib.pyplot as plt

import thesis.subsections.miscellaneous as miscellaneous
import thesis.subsections.equilibrium as equilibrium
import thesis.subsections.sse as sse
import thesis.subsections.sne as sne


def main():
    miscellaneous.main()
    equilibrium.main()
    sse.main()
    sne.main()


if __name__ == '__main__':
    main()
