import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pyphi
import pyphi.utils
import pyphi.visualize
from tqdm.auto import tqdm

_BU_MICRO_SAVEDIR = "results/bu_micro"  # "Binary units micro"
_BBX_MACRO_SAVEDIR = "results/bbx_macro"  # "Blackbox macro"
_BBX_MICRO_SAVEDIR = "results/bbx_micro"  # "Blackbox micro"
_CG_MICRO_SAVEDIR = "results/cg_micro"  # "Coarsegrain micro"
_CG_MACRO_SAVEDIR = "results/cg_macro"  # "Coarsegrain macro"
_MIN_MICRO_SAVEDIR = "results/min_micro"  # "Minimal micro"
_MIN_MACRO_SAVEDIR = "results/min_macro"  # "Minimal macro"
_SFN_MICRO_SAVEDIR = "results/sfn_micro"  # "Something from nothing"
_SFNN_MICRO_SAVEDIR = "results/sfnn_micro"  # "Something from nearly nothing"
_SFS_MICRO_SAVEDIR = "results/sfs_micro"  # "Something from something"


def get_subsets_by_size(network):
    return {
        size: list(
            pyphi.utils.powerset(network.node_indices, min_size=size, max_size=size)
        )
        for size in range(1, len(network.node_indices) + 1)
    }


def get_subsystem_string(network, subset):
    return "".join(sorted(np.array(network.node_labels)[list(subset)].tolist()))


def run_example(
    network,
    network_state,
    subsystem_size=None,
    subsystem_index=None,
    verbose=1,
    savedir=None,
):
    # Create savedir if it does not already exist
    if savedir is not None:
        savedir = pathlib.Path(savedir)
        savedir.mkdir(parents=True, exist_ok=True)

    # Get specific subsystems to check
    if subsystem_size is not None:
        subsets = get_subsets_by_size(network)[subsystem_size]
        if subsystem_index is not None:
            subsets = [subsets[subsystem_index]]
    else:  # Check all subsystems
        subsets = list(
            pyphi.utils.powerset(network.node_indices, nonempty=True, reverse=True)
        )

    sias = {}
    for subset in tqdm(subsets):
        # Print progress for user
        subsystem_string = get_subsystem_string(network, subset)
        if verbose:
            print(f"Doing {subsystem_string}...")

        # Do actual work
        subsystem = pyphi.Subsystem(network, network_state, subset)
        sias[subset] = subsystem.sia()

        # Print output and save
        if verbose == 1:
            print(f"    φ_s: {[sias[subset].phi]}")
        elif verbose >= 2:
            print(sias[subset])
        if savedir is not None:
            with open(savedir / f"{subsystem_string}.pickle", "wb") as f:
                pickle.dump(sias[subset], f, protocol=pickle.HIGHEST_PROTOCOL)


def summarize_example(network, savedir):
    savedir = pathlib.Path(savedir)
    subsets = get_subsets_by_size(network)
    with open(savedir / "summary.txt", "w") as f:
        for subsystem_size in range(1, len(network.node_indices) + 1):
            f.write(f"====={subsystem_size}-node subsystems=====\n")
            for subset in subsets[subsystem_size]:
                subsystem_string = get_subsystem_string(network, subset)
                subsystem_file = savedir / f"{subsystem_string}.pickle"
                if subsystem_file.exists():
                    with open(subsystem_file, "rb") as pf:
                        sia = pickle.load(pf)
                    f.write(f"φ_s({subsystem_string}) = {sia.phi}\n")


def run_binary_units_micro_example(savedir=_BU_MICRO_SAVEDIR, **kwargs):
    network, network_state = get_binary_units_micro_example()
    run_example(network, network_state, savedir=savedir, **kwargs)


def summarize_binary_units_micro_example(savedir=_BU_MICRO_SAVEDIR):
    network, _ = get_binary_units_micro_example()
    summarize_example(network, savedir)


def run_blackbox_micro_example(savedir=_BBX_MICRO_SAVEDIR, **kwargs):
    network, network_state = get_blackbox_micro_example()
    run_example(network, network_state, savedir=savedir, **kwargs)


def summarize_blackbox_micro_example(savedir=_BBX_MICRO_SAVEDIR):
    network, _ = get_blackbox_micro_example()
    summarize_example(network, savedir)


def run_blackbox_macro_example(savedir=_BBX_MACRO_SAVEDIR, **kwargs):
    network, network_state = get_blackbox_macro_example()
    run_example(network, network_state, savedir=savedir, **kwargs)


def summarize_blackbox_macro_example(savedir=_BBX_MACRO_SAVEDIR):
    network, _ = get_blackbox_macro_example()
    summarize_example(network, savedir)


def run_coarsegrain_micro_example(savedir=_CG_MICRO_SAVEDIR, **kwargs):
    network, network_state = get_coarsegrain_micro_example()
    run_example(network, network_state, savedir=savedir, **kwargs)


def summarize_coarsegrain_micro_example(savedir=_CG_MICRO_SAVEDIR):
    network, _ = get_coarsegrain_micro_example()
    summarize_example(network, savedir)


def run_coarsegrain_macro_example(savedir=_CG_MACRO_SAVEDIR, **kwargs):
    network, network_state = get_coarsegrain_macro_example()
    run_example(network, network_state, savedir=savedir, **kwargs)


def summarize_coarsegrain_macro_example(savedir=_CG_MACRO_SAVEDIR):
    network, _ = get_coarsegrain_macro_example()
    summarize_example(network, savedir)


def run_minimal_micro_example(savedir=_MIN_MICRO_SAVEDIR, **kwargs):
    network, network_state = get_minimal_micro_example()
    run_example(network, network_state, savedir=savedir, **kwargs)


def summarize_minimal_micro_example(savedir=_MIN_MICRO_SAVEDIR):
    network, _ = get_minimal_micro_example()
    summarize_example(network, savedir)


def run_minimal_macro_example(savedir=_MIN_MACRO_SAVEDIR, **kwargs):
    network, network_state = get_minimal_macro_example()
    run_example(network, network_state, savedir=savedir, **kwargs)


def summarize_minimal_macro_example(savedir=_MIN_MACRO_SAVEDIR):
    network, _ = get_minimal_macro_example()
    summarize_example(network, savedir)


def run_something_from_nothing_micro_example(savedir=_SFN_MICRO_SAVEDIR, **kwargs):
    network, network_state = get_something_from_nothing_micro_example()
    run_example(network, network_state, savedir=savedir, **kwargs)


def summarize_something_from_nothing_micro_example(savedir=_SFN_MICRO_SAVEDIR):
    network, _ = get_something_from_nothing_micro_example()
    summarize_example(network, savedir)


def run_something_from_nearly_nothing_micro_example(
    savedir=_SFNN_MICRO_SAVEDIR, **kwargs
):
    network, network_state = get_something_from_nearly_nothing_micro_example()
    run_example(network, network_state, savedir=savedir, **kwargs)


def summarize_something_from_nearly_nothing_micro_example(savedir=_SFNN_MICRO_SAVEDIR):
    network, _ = get_something_from_nearly_nothing_micro_example()
    summarize_example(network, savedir)


def run_something_from_something_micro_example(savedir=_SFS_MICRO_SAVEDIR, **kwargs):
    network, network_state = get_something_from_something_micro_example()
    run_example(network, network_state, savedir=savedir, **kwargs)


def summarize_something_from_something_micro_example(savedir=_SFS_MICRO_SAVEDIR):
    network, _ = get_something_from_something_micro_example()
    summarize_example(network, savedir)


def _get_iit4_fig6d_micro_tpm():
    node_labels = ("A", "B", "C", "D", "E", "F")
    network_size = len(node_labels)
    current_states = np.array(list(pyphi.utils.all_states(network_size)))
    tpm = np.zeros_like(current_states, dtype=float)

    k = 4
    for current_state, p in zip(current_states, tpm):
        # Convert 0s to -1s for sigmoidal activation function
        current_state = np.where(current_state == 0, -1, current_state)
        total_input = np.sum(current_state)
        prob = 1.0 / (1.0 + np.exp(-k * total_input))
        p[:] = prob

    return tpm, current_states, node_labels


def get_iit4_fig6d():
    tpm, _, node_labels = _get_iit4_fig6d_micro_tpm()
    network = pyphi.Network(tpm, node_labels=node_labels)
    state = (1, 0, 0, 0, 0, 0)
    return network, state


def get_binary_units_micro_example():
    tpm = np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [0, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 0, 0],
        ]
    )
    node_labels = ("A", "B", "C")
    network = pyphi.Network(tpm, node_labels=node_labels)
    state = (0, 0, 0)
    return network, state


def _get_blackbox_example_micro_tpm():
    node_labels = ("A", "B", "C", "D", "E", "F", "G", "H")
    network_size = len(node_labels)
    current_states = np.array(list(pyphi.utils.all_states(network_size)))
    tpm = np.zeros_like(current_states, dtype=float)

    for current_state, p in zip(current_states, tpm):
        p[0] = (
            0.01
            + 0.01 * current_state[0]
            + 0.1 * current_state[3]
            + 0.8 * current_state[6]
            + 0.05 * current_state[1]
        )  # A
        p[1] = (
            0.01
            + 0.01 * current_state[1]
            + 0.1 * current_state[3]
            + 0.8 * current_state[6]
            + 0.05 * current_state[0]
        )  # B
        p[2] = (
            0.01
            + 0.01 * current_state[2]
            + 0.85 * int(current_state[0] + current_state[1] > 0)
            + 0.1 * int(current_state[0] + current_state[1] == 2)
        )  # C
        p[3] = (
            0.01
            + 0.01 * current_state[3]
            + 0.85 * current_state[2]
            + 0.05 * (current_state[0] + current_state[1])
        )  # D
        p[4] = (
            0.01
            + 0.01 * current_state[4]
            + 0.1 * current_state[7]
            + 0.8 * current_state[2]
            + 0.05 * current_state[5]
        )  # E
        p[5] = (
            0.01
            + 0.01 * current_state[5]
            + 0.1 * current_state[7]
            + 0.8 * current_state[2]
            + 0.05 * current_state[4]
        )  # F
        p[6] = (
            0.01
            + 0.01 * current_state[6]
            + 0.85 * int(current_state[4] + current_state[5] > 0)
            + 0.1 * int(current_state[4] + current_state[5] == 2)
        )  # G
        p[7] = (
            0.01
            + 0.01 * current_state[7]
            + 0.85 * current_state[6]
            + 0.05 * (current_state[4] + current_state[5])
        )  # H

    return tpm, current_states, node_labels


def get_blackbox_micro_example():
    tpm, _, node_labels = _get_blackbox_example_micro_tpm()
    network = pyphi.Network(tpm, node_labels=node_labels)
    state = (1, 1, 1, 1, 1, 1, 1, 1)
    return network, state


def _get_blackbox_example_macro_tpm():
    micro_tpm, micro_states, _ = _get_blackbox_example_micro_tpm()

    tpm = pyphi.convert.sbn2sbs(micro_tpm)
    tpm2 = np.dot(tpm, tpm)  # Take tau=2

    D00 = np.where((micro_states[:, 2] == 0) & (micro_states[:, 6] == 0))[0]
    D10 = np.where((micro_states[:, 2] == 1) & (micro_states[:, 6] == 0))[0]
    D01 = np.where((micro_states[:, 2] == 0) & (micro_states[:, 6] == 1))[0]
    D11 = np.where((micro_states[:, 2] == 1) & (micro_states[:, 6] == 1))[0]

    Dalpha0 = np.where(micro_states[:, 2] == 0)[0]
    Dalpha1 = np.where(micro_states[:, 2] == 1)[0]

    Dbeta0 = np.where(micro_states[:, 6] == 0)[0]
    Dbeta1 = np.where(micro_states[:, 6] == 1)[0]

    assert micro_tpm.shape[0] == 2**8
    assert micro_tpm.shape[1] == 8
    macro_network_size = 2
    macro_states = np.array(list(pyphi.utils.all_states(macro_network_size)))
    macro_tpm = np.zeros_like(macro_states, dtype=float)  # state-by-node
    macro_node_labels = ("α", "β")

    macro_tpm[0, 0] = np.mean(np.sum(tpm2[D00[:, np.newaxis], Dalpha1], axis=1))
    macro_tpm[0, 1] = np.mean(np.sum(tpm2[D00[:, np.newaxis], Dbeta1], axis=1))

    macro_tpm[1, 0] = np.mean(np.sum(tpm2[D10[:, np.newaxis], Dalpha1], axis=1))
    macro_tpm[1, 1] = np.mean(np.sum(tpm2[D10[:, np.newaxis], Dbeta1], axis=1))

    macro_tpm[2, 0] = np.mean(np.sum(tpm2[D01[:, np.newaxis], Dalpha1], axis=1))
    macro_tpm[2, 1] = np.mean(np.sum(tpm2[D01[:, np.newaxis], Dbeta1], axis=1))

    macro_tpm[3, 0] = np.mean(np.sum(tpm2[D11[:, np.newaxis], Dalpha1], axis=1))
    macro_tpm[3, 1] = np.mean(np.sum(tpm2[D11[:, np.newaxis], Dbeta1], axis=1))

    return macro_tpm, macro_states, macro_node_labels


def get_blackbox_macro_example():
    tpm, _, node_labels = _get_blackbox_example_macro_tpm()
    network = pyphi.Network(tpm, node_labels=node_labels)
    state = (1, 1)
    return network, state


def get_coarsegrain_micro_example():
    tpm = np.array(
        [
            [0.05, 0.05, 0.05, 0.05],
            [0.06, 0.15, 0.05, 0.05],
            [0.15, 0.06, 0.05, 0.05],
            [0.16, 0.16, 0.85, 0.85],
            [0.05, 0.05, 0.06, 0.15],
            [0.06, 0.15, 0.06, 0.15],
            [0.15, 0.06, 0.06, 0.15],
            [0.16, 0.16, 0.86, 0.95],
            [0.05, 0.05, 0.15, 0.06],
            [0.06, 0.15, 0.15, 0.06],
            [0.15, 0.06, 0.15, 0.06],
            [0.16, 0.16, 0.95, 0.86],
            [0.85, 0.85, 0.16, 0.16],
            [0.86, 0.95, 0.16, 0.16],
            [0.95, 0.86, 0.16, 0.16],
            [0.96, 0.96, 0.96, 0.96],
        ]
    )

    node_labels = ("A", "B", "C", "D")
    network = pyphi.Network(tpm, node_labels=node_labels)
    state = (0, 0, 0, 0)
    return network, state


def get_coarsegrain_macro_example():
    tpm = np.array(
        [[0.006833, 0.006833], [0.0256, 0.7855], [0.7855, 0.0256], [0.9212, 0.9212]]
    )
    node_labels = ("α", "β")
    network = pyphi.Network(tpm, node_labels=node_labels)
    state = (0, 0)
    return network, state


def get_minimal_micro_example():
    tpm = np.array(
        [
            [0.05, 0.05],
            [0.05, 0.06],
            [0.06, 0.05],
            [0.95, 0.95],
        ]
    )

    node_labels = ("A", "B")
    network = pyphi.Network(tpm, node_labels=node_labels)
    state = (0, 0)
    return network, state


def get_minimal_macro_example():
    tpm = np.array([[0.05 * 0.05 + 2 * 0.01 * 0.05 / 3], [(1 - 0.05) * (1 - 0.05)]])
    node_labels = ("α",)
    network = pyphi.Network(tpm, node_labels=node_labels)
    state = (0,)
    return network, state


def get_dancing_couple_node(
    current_state: np.ndarray,
    self_index: int,
    horizontal_neighbor: int,
    vertical_neighbor: int,
    w_vertical: float,
    w_base: float = 0.05,
    w_self: float = 0.05,
    w_horizontal: float = 0.6,
) -> float:
    return (
        w_base
        + w_self * current_state[self_index]
        + w_horizontal * current_state[horizontal_neighbor]
        + w_vertical * current_state[vertical_neighbor]
    )


def get_dancing_couples_network(w_vertical: float) -> np.ndarray:
    node_labels = ("A", "B", "C", "D")
    connectivity = {
        0: {"horizontal_neighbor": 1, "vertical_neighbor": 2},
        1: {"horizontal_neighbor": 0, "vertical_neighbor": 3},
        2: {"horizontal_neighbor": 3, "vertical_neighbor": 0},
        3: {"horizontal_neighbor": 2, "vertical_neighbor": 1},
    }
    network_size = len(node_labels)
    current_states = np.array(list(pyphi.utils.all_states(network_size)))
    tpm = np.zeros_like(current_states, dtype=float)
    for current_state, p in zip(current_states, tpm):
        for i in connectivity:
            p[i] = get_dancing_couple_node(
                current_state,
                i,
                connectivity[i]["horizontal_neighbor"],
                connectivity[i]["vertical_neighbor"],
                w_vertical=w_vertical,
            )
    return pyphi.Network(tpm, node_labels=node_labels)


def get_something_from_nothing_micro_example():
    network = get_dancing_couples_network(w_vertical=0.00)
    state = (0, 0, 0, 0)
    return network, state


def get_something_from_nearly_nothing_micro_example():
    network = get_dancing_couples_network(w_vertical=0.01)
    state = (0, 0, 0, 0)
    return network, state


def get_something_from_something_micro_example():
    network = get_dancing_couples_network(w_vertical=0.25)
    state = (0, 0, 0, 0)
    return network, state


#### Plotting functions #####


def plot_sbs_tpm(network, use_node_labels=True, height=None):
    def _italicize(text):
        return "$\it{" + "".join(text) + "}$"

    sbs = pyphi.convert.state_by_node2state_by_state(network.tpm)

    states_labels = list(pyphi.utils.all_states(network.size))
    if use_node_labels:
        state_labels = [
            pyphi.visualize.phi_structure.text.Labeler(
                state, network.node_labels, postprocessor=_italicize
            ).nodes(network.node_indices)
            for state in states_labels
        ]

    figsize = None if height is None else (height, height)
    fig, ax = plt.subplots(figsize=figsize)
    ax.pcolormesh(sbs, edgecolors="k", linewidth=0.5, cmap="Greys", vmin=0, vmax=1)
    ax.tick_params(
        top=False,
        labeltop=use_node_labels,
        bottom=False,
        labelbottom=False,
        left=False,
        labelleft=use_node_labels,
    )
    if use_node_labels:
        ax.set_xticks(
            np.arange(len(state_labels)) + 0.5, labels=state_labels, rotation=90
        )
        ax.set_yticks(np.arange(len(state_labels)) + 0.5, labels=state_labels)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    return fig, ax
