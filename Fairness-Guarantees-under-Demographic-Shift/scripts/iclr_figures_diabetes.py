import numpy as np
import os
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import hex2color

from helpers.io import SMLAResultsReader
from helpers.argsweep import ArgumentSweeper
import datasets.diabetes.diabetes as diabetes

if __name__ == '__main__':
    with ArgumentSweeper() as parser:
        parser.add_argument('--unknown_ds',  action='store_true',
                            help='Generates results for the unknown_ds experiments.')
        parser.add_argument('--dshift_var', default='sex',
                            help='Generates results for the dshift variable experiments.')
        parser.add_argument('--print_stats',  action='store_true',
                            help='print accuracy results for the experiments.')
        args = parser.parse_args()

    # Whether to save figures. If false, figures are displayed only.
    save_figs = True

    # Location to save the figures (will be created if nonexistent)
    mode = 'antag' if args.unknown_ds else 'fixed'
    figpath = f'figures/iclr_{mode}_demographic_shift_diabetes_rl'

    # Figure format
    fmt = 'png'

    # Figure DPI for raster formats
    dpi = 200

    # Paths to results files. Figures will be skipped if data cannot be found.
    di_path = f'results/results_diabetes_experiments/iclr_diabetes_{mode}_ds_rl_{args.dshift_var}_di_0/iclr_diabetes_{mode}_ds_rl_{args.dshift_var}_di.h5'
    dp_path = f'results/results_diabetes_experiments/iclr_diabetes_{mode}_ds_rl_{args.dshift_var}_dp_0/iclr_diabetes_{mode}_ds_rl_{args.dshift_var}_dp.h5'

    all_paths = {
        'di': di_path,
        'dp': dp_path,
    }

    def thousands_fmt(x, pos):
        return f'{x/10**3:,.0f}K'

    def percentage_fmt(x, pos):
        return f'{x:,.1f}%'

    ThousandsFmt = mtick.FuncFormatter(thousands_fmt)
    PercentageFmt = mtick.FuncFormatter(percentage_fmt)

    # Value of delta used in experiments
    delta = 0.05

    # Constants for rendering figures
    if args.dshift_var == 'sex':
        n_total = diabetes.load_s(R0='Black', R1='White').training_splits()[
            'X'].shape[0]
    elif args.dshift_var == 'race':
        n_total = diabetes.load_r().training_splits()['X'].shape[0]

    # Mapping from model names that will appear on legends
    pprint_map = {
        'SC': 'Seldonian',
        'QSC': 'Quasi-Seldonian',
        'QSRC': 'Shifty',
        'FairlearnSVC': 'Fairlearn',
        'FairConst': 'Fairness Constraints',
        'FairRobust': 'RFLearn'
    }

    legend_priority = {
        'Seldonian': 0,
        'Quasi-Seldonian': -0.5,
        'Shifty': 0.5,
        'Fairness Constraints': -0.7,
        'Fairlearn': -0.71,
        'RFLearn': 0.1,
    }

    standard_smla_names = ['SC', 'QSC']
    robust_smla_names = ['QSRC']

    keep_mname_list = ['SC', 'QSC', 'QSRC',
                       'FairConst', 'FairlearnSVC', 'FairRobust']

    # Create the figure directory if nonexistent
    if save_figs and not (os.path.isdir(figpath)):
        os.makedirs(figpath)

    #############
    #  Helpers  #
    #############

    def save(fig, path, *args, **kwargs):
        if not (os.path.isdir(figpath)):
            os.makedirs(figpath)
        path = os.path.join(figpath, path)
        print('Saving figure to \'%s\'' % path)
        fig.savefig(path, *args, **kwargs)

    def get_ls(name):
        if name == 'QSC':
            return '--'
        return '-'

    def get_lw(name):
        if name == 'QSRC':
            return 2
        return 1

    def get_diabetes_stats(path, keep_mname_list=keep_mname_list):
        ''' Helper for extracting resutls from diabetes results files. '''
        results_container = SMLAResultsReader(path)
        results_container.open()
        task_parameters = results_container._store['task_parameters']
        results = results_container._store['results']

        n_train = np.array(task_parameters.n_train)
        arates, arates_se = [], []  # Acceptance rates and SEs
        # Failure rates ans SEs (rate that accepted solutions have g(theta) > 0 on the test set)
        ofrates, ofrates_se = [], []
        # Failure rates ans SEs (rate that accepted solutions have g(theta) > 0 on the test set)
        dfrates, dfrates_se = [], []
        olrates, olrates_se = [], []  # Test set error and SEs
        dlrates, dlrates_se = [], []  # Test set error and SEs
        mnames = np.unique(results.name).astype(str)
        mnames = [mn for mn in mnames if mn in keep_mname_list]
        pmnames = np.array([pprint_map[name] for name in mnames])

        for tid, _ in enumerate(n_train):
            _arates, _arates_se = [], []
            _ofrates, _ofrates_se = [], []
            _dfrates, _dfrates_se = [], []
            _olrates, _olrates_se = [], []
            _dlrates, _dlrates_se = [], []
            for mname in mnames:
                _results = results[np.logical_and(
                    results.tid == tid, results.name == mname)]
                if len(np.unique(_results.pid)) > 1:
                    _results = _results[_results.pid == 0]

                accepts = ~np.array(_results.original_nsf)
                # keyboard()
                original_gs = np.array([_g for _g in _results.original_g])
                antagonist_gs = np.array([_g for _g in _results.antagonist_g])
                og = np.array([(g > 0 if a else False)
                              for (g, a) in zip(original_gs, accepts)])
                dg = np.array([(g > 0 if a else False)
                              for (g, a) in zip(antagonist_gs, accepts)])
                oa = np.array(_results.original_acc)
                da = np.array(_results.antagonist_acc)

                n = len(accepts)
                n_accepts = sum(accepts)

                # Record I[ a(D) != NSF ]
                _arates.append(np.mean(accepts))
                _arates_se.append(np.std(accepts, ddof=1) /
                                  np.sqrt(len(accepts)))

                # Estimate Pr( g(a(D)) > 0 )
                _ofrates.append(og.mean())
                _dfrates.append(dg.mean())
                if n > 1:
                    _ofrates_se.append(og.std(ddof=1)/np.sqrt(n))
                    _dfrates_se.append(dg.std(ddof=1)/np.sqrt(n))
                else:
                    _ofrates_se.append(np.nan)
                    _dfrates_se.append(np.nan)

                # Estimate E[ loss | a(D) != NSF ]
                if n_accepts > 1:
                    _olrates.append(oa[accepts].mean())
                    _dlrates.append(da[accepts].mean())
                    _olrates_se.append(oa[accepts].std(
                        ddof=1)/np.sqrt(n_accepts))
                    _dlrates_se.append(da[accepts].std(
                        ddof=1)/np.sqrt(n_accepts))
                elif n_accepts == 1:
                    _olrates.append(oa[accepts].mean())
                    _dlrates.append(da[accepts].mean())
                    _olrates_se.append(np.nan)
                    _dlrates_se.append(np.nan)
                else:
                    _olrates.append(np.nan)
                    _dlrates.append(np.nan)
                    _olrates_se.append(np.nan)
                    _dlrates_se.append(np.nan)

            arates.append(_arates)
            ofrates.append(_ofrates)
            dfrates.append(_dfrates)
            olrates.append(_olrates)
            dlrates.append(_dlrates)
            arates_se.append(_arates_se)
            ofrates_se.append(_ofrates_se)
            dfrates_se.append(_dfrates_se)
            olrates_se.append(_olrates_se)
            dlrates_se.append(_dlrates_se)

        arates = np.array(arates)
        arates_se = np.array(arates_se)
        ofrates = np.array(ofrates)
        ofrates_se = np.array(ofrates_se)
        dfrates = np.array(dfrates)
        dfrates_se = np.array(dfrates_se)
        olrates = np.array(olrates)
        olrates_se = np.array(olrates_se)
        dlrates = np.array(dlrates)
        dlrates_se = np.array(dlrates_se)

        # Assign colors to each method
        # This part is a hack to get reasonable colors for each method. If more methods are
        #   added this section should be changed.
        colors = []
        for nm in mnames:
            if nm in robust_smla_names:
                colors.append(hex2color('#4daf4a'))
            elif nm in standard_smla_names:
                colors.append(hex2color('#377eb8'))
            elif nm == 'FairlearnSVC':
                colors.append(hex2color('#FF9E44'))
            elif nm == 'FairConst':
                colors.append(hex2color('#e41a1c'))
            elif nm == 'FairRobust':
                colors.append(hex2color('#6f32a8'))
            else:
                colors.append(hex2color('#e41a1c'))

        def add_noise(X, e=0.01):
            return X + e*(np.random.random(size=X.shape) - 0.5)
        out = {
            'mnames': mnames,
            'pmnames': pmnames,
            'colors': colors,
            'n_train': n_train,
            'arate_v_n': arates,
            'arate_se_v_n': arates_se,
            'ofrate_v_n': add_noise(ofrates, 0.02),
            'ofrate_se_v_n': ofrates_se,
            'dfrate_v_n': add_noise(dfrates, 0.02),
            'dfrate_se_v_n': dfrates_se,
            'olrate_v_n': 100 * olrates,
            'olrate_se_v_n': 100 * olrates_se,
            'dlrate_v_n': 100 * dlrates,
            'dlrate_se_v_n': 100 * dlrates_se
        }
        return out


def plotting_NSF_acc(path):

    if not (os.path.exists(path)):
        print('No results found at path \'%s\'. Skipped.' % path)
    else:
        D = get_diabetes_stats(path)
        arates = D['arate_v_n']
        arates_se = D['arate_se_v_n']
        mnames = D['mnames']
        colors = D['colors']
        nvals = D['n_train']

        fig, ax_ar = plt.subplots()

        # Plot acceptance rate
        legend_data, added = [], []
        for mn, c, ar, se in zip(mnames[::-1], colors[::-1], (arates.T)[::-1], (arates_se.T)[::-1]):
            line = ax_ar.plot(nvals, (1-ar), c=c,
                              ls=get_ls(mn), lw=get_lw(mn))[0]
            ax_ar.fill_between(nvals, ((1-ar)+se), ((1-ar)-se),
                               alpha=0.25, linewidth=0, color=c)
            pmn = pprint_map[mn]
            if not (pmn in added):
                added.append(pmn)
                legend_data.append(line)
        legend_data, added = legend_data[::-1], added[::-1]
        ax_ar.set_xlabel('Training Samples', labelpad=3.5)
        ax_ar.set_ylabel('Pr(NO_SOLUTION_FOUND)', labelpad=7)
        ax_ar.set_xlim(right=max(nvals))
        ax_ar.set_ylim((0, 1))
        ax_ar.xaxis.set_major_formatter(ThousandsFmt)

        # Finalize the figure and display/save
        ax_ar.spines['right'].set_visible(False)
        ax_ar.spines['top'].set_visible(False)

        if save_figs:
            save(fig, f'icrl_diabetes_{mode}_ds_rl_{args.dshift_var}_{path[-5:-3]}_P(NSF).{fmt}', dpi=dpi)
        else:
            fig.show()

        # Legend figure
        fig = plt.figure(figsize=(10.75, 0.3))
        priorities = [legend_priority[n] for n in added]
        added = [added[i] for i in np.argsort(priorities)[::-1]]
        legend_data = [legend_data[i] for i in np.argsort(priorities)[::-1]]
        fig.legend(legend_data, added, loc='center', fancybox=True, ncol=len(
            legend_data), columnspacing=1, fontsize=11, handletextpad=0.5)
        save(fig, 'iclr_legend.%s' % fmt, dpi=dpi)

        if args.print_stats:
            # Numerical stats
            print(mode, path[-5:-3])

            oacc_v_n = D['olrate_v_n']
            dacc_v_n = D['dlrate_v_n']

            # Accuracy Original
            print('\nAccuracy (Original)\n')
            for mn, c, acc in zip(mnames[::-1], colors[::-1], (oacc_v_n.T)[::-1]):
                string = mn + ':\n' + "{:.2f} ".format(acc[0])
                for a in acc[1:]:
                    string += "& {:.2f} ".format(a)
                print(string)

            # Accuracy Deployed
            print('\nAccuracy (Deployed)\n')
            for mn, c, acc in zip(mnames[::-1], colors[::-1], (dacc_v_n.T)[::-1]):
                string = mn + ':\n' + "{:.2f} ".format(acc[0])
                for a in acc[1:]:
                    string += "& {:.2f} ".format(a)
                print(string)

            print('\nAccuracy Difference\n')
            for mn, acco, accd in zip(mnames[::-1], (oacc_v_n.T)[::-1], (dacc_v_n.T)[::-1]):
                diff = np.round(accd, 2)-np.round(acco, 2)
                string = mn + ':\n' + "{:.2f} ".format(diff[0])
                for a in diff[1:]:
                    string += "& {:.2f} ".format(a)
                print(string)

for path in all_paths:
    plotting_NSF_acc(all_paths[path])