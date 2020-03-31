import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from info_metrics import gaussian_dataset
from info_metrics.information_estimator_v2 import InformationEstimator


def set_axis_color(ax, color='#212121'):
    ax.spines['bottom'].set_color(color)
    ax.spines['top'].set_color(color)
    ax.spines['right'].set_color(color)
    ax.spines['left'].set_color(color)
    ax.tick_params(axis='x', colors=color)
    ax.tick_params(axis='y', colors=color)
    ax.yaxis.label.set_color(color)
    ax.xaxis.label.set_color(color)
    ax.title.set_color(color)
    return ax


def set_legend_color(lg, color='#212121'):
    plt.setp(lg.get_texts(), color=color)
    return lg


if __name__ == '__main__':
    # Settings of experiment
    show_shannon = True
    show_comparison = True
    show_batch_change = True

    n_tries = 50
    n_corr = 19  # 19 is full, 10 is short
    dimension_list = [10, 100, 1000]
    n_samples_batch_list = [128, 256, 512]
    sigma_zero = 2.0
    n_samples = 128
    batch_change_dims = 100
    old_color = '#424242'
    new_color = '#0277bd'
    alpha = 0.3
    markersize = 2
    res_mi = 2

    corr_factor_list = np.linspace(-0.9, 0.9, n_corr)
    factor_mi = 10 ** res_mi

    if show_shannon:
        # Shannon case d=1
        mut_info_list = []
        for corr_factor in corr_factor_list:
            mut_info = gaussian_dataset.get_shannon_mut_info(1, corr_factor)
            mut_info_list.append(mut_info)
        # Plots of Shannon
        fig, ax = plt.subplots(1, 1, figsize=(3, 2))
        ax.plot(
            corr_factor_list, mut_info_list,
            linewidth=1, marker='o', markersize=markersize,
            color=old_color)
        ax.set_ylabel('I(X;Y)', fontsize=8)
        ax.yaxis.labelpad = -9
        ax.set_ylim([0, 0.9])
        ax.set_yticks([0, 0.9])
        ax.set_xticks([-1, 0, 1])
        ax.set_xlim([-1, 1])
        ax.tick_params(labelsize=8)
        ax = set_axis_color(ax)
        ax.set_title('Theoretical MI for d = 1', fontsize=8)
        ax.set_xlabel('Correlation Factor', fontsize=8)
        plt.tight_layout()

        plt.savefig(
            "shannon.pdf", pad_inches=0.5)
        plt.show()

    if show_comparison:
        # Comparison
        fig, ax = plt.subplots(len(dimension_list), 2, figsize=(4, 4), sharex=True)

        for i, dimension in enumerate(dimension_list):

            print('\nDimension %d' % dimension)

            # Estimation old rule
            tf.reset_default_graph()
            estimator = InformationEstimator(
                sigma_zero * np.sqrt(dimension_list[0]), normalize_dimension=False)
            corr_factor_ph = tf.placeholder(tf.float32, shape=())
            x_samples, y_samples = gaussian_dataset.generate_batch_tf(
                dimension, corr_factor_ph, n_samples)
            mi_estimation_tf = estimator.mutual_information(x_samples, y_samples)
            sess = tf.Session()
            mean_estimation_list = []
            std_estimation_list = []
            for corr_factor in corr_factor_list:
                print('Correlation factor %1.1f' % corr_factor)
                estimation_tries_list = []
                for i_try in range(n_tries):
                    mi_estimation_np = sess.run(
                        mi_estimation_tf,
                        feed_dict={corr_factor_ph: corr_factor})
                    estimation_tries_list.append(mi_estimation_np)
                mean_estimation_list.append(np.mean(estimation_tries_list))
                std_estimation_list.append(np.std(estimation_tries_list))
            mean_estimation_list = np.array(mean_estimation_list)
            std_estimation_list = np.array(std_estimation_list)

            # Add figure with results
            ax[i, 0].plot(
                corr_factor_list, mean_estimation_list,
                label='d = %d' % dimension,
                linewidth=1, marker='o', markersize=markersize,
                color=old_color
            )
            ax[i, 0].fill_between(
                corr_factor_list,
                mean_estimation_list - std_estimation_list,
                mean_estimation_list + std_estimation_list,
                color=old_color, alpha=alpha
            )

            ax[i, 0].set_ylabel('I(X;Y)', fontsize=8)
            ax[i, 0].yaxis.labelpad = -9
            max_display = np.max(mean_estimation_list + std_estimation_list)
            min_display = np.min(mean_estimation_list - std_estimation_list)
            max_display = np.ceil(factor_mi * max_display) / factor_mi
            min_display = np.floor(factor_mi * min_display) / factor_mi
            # max_display = np.round(max_display + delta_mi, decimals=2)
            # min_display = np.round(min_display - delta_mi, decimals=2)
            delta_to_add = 0.1 * (max_display - min_display)
            ax[i, 0].set_ylim([min_display - delta_to_add, max_display + delta_to_add])
            ax[i, 0].set_yticks([min_display, max_display])
            ax[i, 0].set_title('d = %d' % dimension, fontsize=8)

            list_of_mins = [min_display]
            list_of_max = [max_display]

            # Estimation new rule
            tf.reset_default_graph()
            estimator = InformationEstimator(sigma_zero, normalize_dimension=True)
            corr_factor_ph = tf.placeholder(tf.float32, shape=())
            x_samples, y_samples = gaussian_dataset.generate_batch_tf(
                dimension, corr_factor_ph, n_samples)
            mi_estimation_tf = estimator.mutual_information(x_samples, y_samples)
            sess = tf.Session()
            mean_estimation_list = []
            std_estimation_list = []
            for corr_factor in corr_factor_list:
                print('Correlation factor %1.1f' % corr_factor)
                estimation_tries_list = []
                for i_try in range(n_tries):
                    mi_estimation_np = sess.run(
                        mi_estimation_tf,
                        feed_dict={corr_factor_ph: corr_factor})
                    estimation_tries_list.append(mi_estimation_np)
                mean_estimation_list.append(np.mean(estimation_tries_list))
                std_estimation_list.append(np.std(estimation_tries_list))
            mean_estimation_list = np.array(mean_estimation_list)
            std_estimation_list = np.array(std_estimation_list)

            # Add figure with results
            ax[i, 1].plot(
                corr_factor_list, mean_estimation_list,
                label='d = %d' % dimension,
                linewidth=1, marker='o', markersize=markersize,
                color=new_color
            )
            ax[i, 1].fill_between(
                corr_factor_list,
                mean_estimation_list - std_estimation_list,
                mean_estimation_list + std_estimation_list,
                color=new_color, alpha=alpha
            )
            max_display = np.max(mean_estimation_list + std_estimation_list)
            min_display = np.min(mean_estimation_list - std_estimation_list)
            # max_display = np.ceil((max_display + delta_mi) * 100) / 100
            # min_display = np.floor((min_display - delta_mi) * 100) / 100
            # max_display = np.round(max_display + delta_mi, decimals=2)
            # min_display = np.round(min_display - delta_mi, decimals=2)
            max_display = np.ceil(factor_mi * max_display) / factor_mi
            min_display = np.floor(factor_mi * min_display) / factor_mi
            delta_to_add = 0.1 * (max_display - min_display)
            ax[i, 1].set_ylim([min_display - delta_to_add, max_display + delta_to_add])
            ax[i, 1].set_yticks([min_display, max_display])
            ax[i, 1].set_title('d = %d' % dimension, fontsize=8)

            list_of_mins.append(min_display)
            list_of_max.append(max_display)
            if i == 0:
                min_display = np.min(min_display)
                max_display = np.max(max_display)
                ax[i, 0].set_ylim([min_display - delta_to_add, max_display + delta_to_add])
                ax[i, 0].set_yticks([min_display, max_display])
                ax[i, 1].set_ylim([min_display - delta_to_add, max_display + delta_to_add])
                ax[i, 1].set_yticks([min_display, max_display])

        for t_ax in ax:
            for s_ax in t_ax:
                # lg = s_ax.legend(loc='upper center', fontsize=8, frameon=False)
                s_ax.set_xticks([-1, 0, 1])
                s_ax.set_xlim([-1, 1])
                s_ax.tick_params(labelsize=8)
                s_ax = set_axis_color(s_ax)
                # lg = set_legend_color(lg)

        # ax[0, 0].set_title('Previous Rule', fontsize=8)
        # ax[0, 1].set_title('Proposed Rule', fontsize=8)
        ax[-1, 0].set_xlabel('Correlation Factor', fontsize=8)
        ax[-1, 1].set_xlabel('Correlation Factor', fontsize=8)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        plt.text(
            x=0.19, y=0.96, fontsize=9,
            s='Previous Rule',
            ha='left', transform=fig.transFigure)
        plt.text(
            x=0.67, y=0.96, fontsize=9,
            s='Proposed Rule',
            ha='left', transform=fig.transFigure)

        plt.text(
            x=0.12, y=0.91, fontsize=16,
            s=r"$\bf{a}$",
            ha="left", transform=fig.transFigure)
        plt.text(
            x=0.6, y=0.91, fontsize=16,
            s=r"$\bf{b}$",
            ha="left", transform=fig.transFigure)

        plt.text(
            x=0.12, y=0.62, fontsize=16,
            s=r"$\bf{c}$",
            ha="left", transform=fig.transFigure)
        plt.text(
            x=0.6, y=0.62, fontsize=16,
            s=r"$\bf{d}$",
            ha="left", transform=fig.transFigure)

        plt.text(
            x=0.12, y=0.335, fontsize=16,
            s=r"$\bf{e}$",
            ha="left", transform=fig.transFigure)
        plt.text(
            x=0.6, y=0.335, fontsize=16,
            s=r"$\bf{f}$",
            ha="left", transform=fig.transFigure)

        plt.savefig(
            "comparison_rule.pdf", pad_inches=0.5)

        plt.show()

    if show_batch_change:
        mins_list = []
        maxes_list = []
        # Comparison
        fig, ax = plt.subplots(1, 3, figsize=(4.5, 1.8), sharex=True, sharey=True)
        dimension = batch_change_dims
        for i, n_samples in enumerate(n_samples_batch_list):
            print('Batch size %d' % n_samples)
            # Estimation new rule
            tf.reset_default_graph()
            estimator = InformationEstimator(
                sigma_zero, normalize_dimension=True)
            corr_factor_ph = tf.placeholder(tf.float32, shape=())
            x_samples, y_samples = gaussian_dataset.generate_batch_tf(
                dimension, corr_factor_ph, n_samples)
            mi_estimation_tf = estimator.mutual_information(
                x_samples, y_samples)
            sess = tf.Session()
            mean_estimation_list = []
            std_estimation_list = []
            for corr_factor in corr_factor_list:
                print('Correlation factor %1.1f' % corr_factor)
                estimation_tries_list = []
                for i_try in range(n_tries):
                    mi_estimation_np = sess.run(
                        mi_estimation_tf,
                        feed_dict={corr_factor_ph: corr_factor})
                    estimation_tries_list.append(mi_estimation_np)
                mean_estimation_list.append(np.mean(estimation_tries_list))
                std_estimation_list.append(np.std(estimation_tries_list))
            mean_estimation_list = np.array(mean_estimation_list)
            std_estimation_list = np.array(std_estimation_list)

            # Add figure with results
            ax[i].plot(
                corr_factor_list, mean_estimation_list,
                label='d = %d' % dimension,
                linewidth=1, marker='o', markersize=markersize,
                color=new_color
            )
            ax[i].fill_between(
                corr_factor_list,
                mean_estimation_list - std_estimation_list,
                mean_estimation_list + std_estimation_list,
                color=new_color, alpha=alpha
            )
            max_display = np.max(mean_estimation_list + std_estimation_list)
            min_display = np.min(mean_estimation_list - std_estimation_list)
            # max_display = np.ceil((max_display + delta_mi) * 100) / 100
            # min_display = np.floor((min_display - delta_mi) * 100) / 100
            # max_display = np.round(max_display + delta_mi, decimals=2)
            # min_display = np.round(min_display - delta_mi, decimals=2)
            max_display = np.ceil(factor_mi * max_display) / factor_mi
            min_display = np.floor(factor_mi * min_display) / factor_mi
            delta_to_add = 0.1 * (max_display - min_display)

            ax[i].set_title('%d samples' % n_samples, fontsize=8)
            ax[i].set_xticks([-1, 0, 1])
            ax[i].set_xlim([-1, 1])
            ax[i].tick_params(labelsize=8)
            ax[i] = set_axis_color(ax[i])
            ax[i].set_xlabel('Correlation Factor', fontsize=8)
            mins_list.append(min_display)
            maxes_list.append(max_display)

        min_display = np.min(mins_list)
        max_display = np.max(maxes_list)
        delta_to_add = 0.1 * (max_display - min_display)
        for s_ax in ax:
            s_ax.set_ylim(
                [min_display - delta_to_add, max_display + delta_to_add])
            s_ax.set_yticks([min_display, max_display])

        ax[0].set_ylabel('I(X;Y)', fontsize=8)
        ax[0].yaxis.labelpad = -9

        plt.tight_layout()

        plt.text(
            x=0.105, y=0.85, fontsize=16,
            s=r"$\bf{a}$",
            ha="left", transform=fig.transFigure)
        plt.text(
            x=0.405, y=0.85, fontsize=16,
            s=r"$\bf{b}$",
            ha="left", transform=fig.transFigure)
        plt.text(
            x=0.71, y=0.85, fontsize=16,
            s=r"$\bf{c}$",
            ha="left", transform=fig.transFigure)

        plt.savefig(
            "comparison_batch.pdf", pad_inches=0.5)

        plt.show()

