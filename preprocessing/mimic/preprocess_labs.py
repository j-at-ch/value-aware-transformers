import os

import numpy as np
import pandas as pd
import pickle as pickle
import tqdm
import shutil

from pprint import pprint
from sklearn.model_selection import train_test_split
from preprocessing_arguments import PreprocessingArguments


def ts_to_posix(time):
    return pd.Timestamp(time, unit='s').timestamp()


uom_scales = {
    50889: {"mg/L": 1, "mg/dL": 10, "MG/DL": 10},
    50916: {"ug/dL": 10, "nG/mL": 1},
    50926: {"mIU/L": 1, "mIU/mL": 1},
    50958: {"mIU/L": 1, "mIU/mL": 1},
    50989: {"pg/mL": 1, "ng/dL": 10},
    51127: {"#/uL": 1, "#/CU MM": 1},
    51128: {"#/uL": 1, "#/CU MM": 1},
}


def unitscale(itemid, valueuom):
    if (itemid in uom_scales) & (valueuom != 'nan'):
        scale_val_by = uom_scales[itemid][valueuom]
    else:
        scale_val_by = 1
    return scale_val_by


def get_numeric_quantile_from_(quantiles_df, itemid, value):
    # maps unknown indices or cats to 1
    # otherwise maps to 2..(num_quantiles+2)
    if itemid not in quantiles_df.index:
        index = -1
    else:
        q = quantiles_df.loc[itemid]  # q is a quants series that may contain nans
        array = (value <= q)
        if value > q.iloc[-1]:
            index = len(q)
        elif not any(array):  # needed for NaN - corresponding to cat variables!
            index = -1
        else:
            a, = np.where(array)
            index = a[0]
    return index + 2


def apply_quantile_fct(labs_df, quantiles_df, col):
    if pd.isna(labs_df[col]):
        return 1
    else:
        return get_numeric_quantile_from_(quantiles_df, labs_df.ITEMID, labs_df[col])


def preprocess_labs_for_1p5D(args):
    print('*' * 17, 'preprocessing labs with the following settings:', sep='\n')
    pprint(vars(args), indent=2)
    print('*' * 17)

    # paths
    d_labitems_path = os.path.join(args.mimic_root, "D_LABITEMS.csv")


    if bool(args.labs_preliminaries_done):
        scaled_labevents_path = os.path.join(args.data_root, "labevents_scaled.csv")
        lab_quantiles_path = os.path.join(args.data_root, "lab_quantiles.csv")
        labevents = (pd.read_csv(scaled_labevents_path,
                                 index_col='ROW_ID',
                                 parse_dates=['CHARTTIME', 'ADMITTIME'])
                     .dropna(subset=['HADM_ID'])
                     .astype({'HADM_ID': 'int', 'VALUEUOM': 'str'})
                     )
    else:
        labevents_path = os.path.join(args.mimic_root, "LABEVENTS.csv")
        labevents = (pd.read_csv(labevents_path,
                                 index_col='ROW_ID',
                                 parse_dates=['CHARTTIME'])
                     .dropna(subset=['HADM_ID'])
                     .astype({'HADM_ID': 'int', 'VALUEUOM': 'str'})
                     )
        lab_quantiles_path = os.path.join(args.save_root, "lab_quantiles.csv")

    if args.augmented_admissions == 'w':
        admissions_path = os.path.join(args.mimic_root, "ADMISSIONS.csv")
        targets_path = os.path.join(args.save_root, "augmented_admissions.csv")

        # read admissions

        admissions = (pd.read_csv(admissions_path,
                                  index_col='HADM_ID',
                                  parse_dates=['ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'EDREGTIME', 'EDOUTTIME'])
                      .drop(['ROW_ID'], axis=1)
                      .rename(columns={'HAS_CHARTEVENTS_DATA': 'HAS_CHARTS'})
                      )

        labsinfo = pd.DataFrame(
            {'NUMLABS': labevents.groupby('HADM_ID').ITEMID.count(),
             'NUMLABVALS': labevents.groupby('HADM_ID').VALUENUM.count(),
             'FIRSTLABTIME': labevents.groupby('HADM_ID').CHARTTIME.min(),
             'LASTLABTIME': labevents.groupby('HADM_ID').CHARTTIME.max()}
        )

        adm = pd.concat([admissions, labsinfo], axis=1)
        adm.loc[:, 'HAS_LABS'] = (~adm.NUMLABS.isna()).astype('int')
        adm.loc[:, 'HADM_IN_SEQ'] = adm.groupby('SUBJECT_ID')['ADMITTIME'].rank().astype(int)
        adm.loc[:, 'LOS'] = (adm.DISCHTIME - adm.ADMITTIME)
        adm.loc[:, 'ADMIT_TO_EXPIRE'] = (adm.DEATHTIME - adm.ADMITTIME)
        adm.loc[:, 'EXPIRE_BEFORE_ADMIT'] = (adm.ADMIT_TO_EXPIRE < pd.Timedelta(days=0)).astype('int')
        adm.loc[:, 'DEATH>1D'] = (adm.ADMIT_TO_EXPIRE > pd.Timedelta(days=1)).astype('int')
        adm.loc[:, 'DEATH>2.5D'] = (adm.ADMIT_TO_EXPIRE > pd.Timedelta(days=2.5)).astype('int')
        adm.loc[:, 'DEATH<=3D'] = (adm.ADMIT_TO_EXPIRE <= pd.Timedelta(days=3)).astype('int')
        adm.loc[:, 'DEATH>3D'] = (adm.ADMIT_TO_EXPIRE > pd.Timedelta(days=3)).astype('int')
        adm.loc[:, 'DEATH<=7D'] = (adm.ADMIT_TO_EXPIRE <= pd.Timedelta(days=7)).astype('int')
        adm.loc[:, 'DEATH>7D'] = (adm.ADMIT_TO_EXPIRE > pd.Timedelta(days=7)).astype('int')
        adm.loc[:, 'DEATH>10D'] = (adm.ADMIT_TO_EXPIRE > pd.Timedelta(days=10)).astype('int')

        # add in calculations relying on a join between ADMISSIONS summaries and LABEVENTS:

        labevents = labevents.join(adm[['ADMITTIME']], on='HADM_ID')
        labevents2 = labevents[labevents.CHARTTIME <= labevents.ADMITTIME + pd.Timedelta(days=2)]

        adm = pd.concat([adm,
                         pd.DataFrame({
                             'NUMLABS<2D': labevents2.groupby('HADM_ID').ITEMID.count(),
                             'NUMLABVALS<2D': labevents2.groupby('HADM_ID').VALUENUM.count(),
                             'NUMLABSUNQ<2D': labevents2.groupby('HADM_ID').ITEMID.nunique()
                         })
                         ], axis=1
                        )

        # select hadms for this data slice:

        qualifying_hadms = adm[(
                (adm.LOS >= pd.Timedelta(days=2)) & (adm['NUMLABS<2D'] >= args.min_num_labs)
                & (pd.isna(adm.ADMIT_TO_EXPIRE) | (adm.ADMIT_TO_EXPIRE >= pd.Timedelta(days=2)))
        )]
        qualifying_hadm_ids = qualifying_hadms.index.to_numpy()

        qualifying_subjects = qualifying_hadms.loc[:, 'SUBJECT_ID'].unique()
        print(qualifying_subjects, len(qualifying_subjects))

        # split qualifying subject_ids into train, val, test and assert that they partition.

        train_subjects, surplus = train_test_split(qualifying_subjects, train_size=0.8, random_state=1965)
        val_subjects, test_subjects = train_test_split(surplus, train_size=0.5, random_state=1965)
        del surplus

        train_indices = qualifying_hadms[qualifying_hadms.SUBJECT_ID.isin(train_subjects)].index.to_numpy()
        val_indices = qualifying_hadms[qualifying_hadms.SUBJECT_ID.isin(val_subjects)].index.to_numpy()
        test_indices = qualifying_hadms[qualifying_hadms.SUBJECT_ID.isin(test_subjects)].index.to_numpy()

        print(f"num_train: {len(train_indices)}",
              f"num_val: {len(val_indices)}",
              f"num_test: {len(test_indices)}",
              f"num_qualifying: {len(qualifying_hadm_ids)}")
        assert set(qualifying_hadm_ids) == set(train_indices) | set(val_indices) | set(test_indices)

        adm.loc[train_indices, 'PARTITION'] = 'train'
        adm.loc[val_indices, 'PARTITION'] = 'val'
        adm.loc[test_indices, 'PARTITION'] = 'test'

        print(f"writing augmented admissions df to {targets_path}...")
        adm.to_csv(targets_path)
        print("written!\n")

    # copy the token dictionary
    shutil.copy(d_labitems_path, os.path.join(args.save_root, 'd_labitems.csv'))
    d_labitems = pd.read_csv(d_labitems_path)

    # generate the tokens:
    special_tokens = {'[PAD]': 0}
    token_shift = len(special_tokens)
    itemid2token = dict(zip(d_labitems['ITEMID'], range(token_shift, token_shift + len(d_labitems))))
    itemid2token.update(special_tokens)
    token2itemid = {v: k for k, v in itemid2token.items()}

    qname2qtoken = dict(
        zip([f'Q{i}' for i in range(1, len(args.quantiles) + 2)],
            range(token_shift + 1, token_shift + 2 + len(args.quantiles)))
    )
    qname2qtoken.update({'UNK': 1})
    qname2qtoken.update(special_tokens)
    qtoken2qname = {v: k for k, v in qname2qtoken.items()}

    assert len(qname2qtoken) == len(args.quantiles) + 3, "num quantile ranges specified + CAT + [PAD] must correspond."

    def map2token(itemid):
        return itemid2token[int(itemid)]

    def get_from_adm(hadm_id, target):
        return adm.loc[hadm_id, target]

    labevents = labevents[labevents.CHARTTIME <= labevents.ADMITTIME + pd.Timedelta(days=2)]

    if not bool(args.labs_preliminaries_done):
        print("unit-scaling lab values...")
        labevents['SCALE'] = labevents.apply(lambda x: unitscale(x['ITEMID'], x['VALUEUOM']), axis=1)
        labevents['VALUE_SCALED'] = labevents['SCALE'] * labevents['VALUENUM']
        #labevents = labevents.join(adm[['ADMITTIME']], on='HADM_ID')
        print("lab values unit-scaled!\n")
        labs_out_path = os.path.join(args.save_root, "labevents_scaled.csv")
        print(f"writing scaled labs df to {labs_out_path} for posterity...")
        labevents.to_csv(labs_out_path)
        print("written!\n")
    else:
        print("lab values are not being rescaled - presumed already rescaled!")

    # loop through index sets and generate output files
    for subset in ['train', 'val', 'test']:
        print(f'Processing {subset} set data...')

        # grouper for labs
        groups = (labevents.query(f'HADM_ID.isin(@{subset}_indices)')
                  .groupby(by='HADM_ID')
                  )

        # train token counts and quantile calculation
        if subset == 'train':
            print("counting train token frequencies...")
            token2trcount = (groups.obj['ITEMID']
                             .apply(map2token)
                             .value_counts()
                             .to_dict()
                             )
            print("train token frequencies counted!\n")

            print("calculating train lab value quants...")
            lab_quantiles_train = groups.obj.groupby('ITEMID').VALUE_SCALED.quantile(args.quantiles)
            print("train lab value quants calculated!\n")
            lab_quantiles_train.to_csv(lab_quantiles_path)
            print(f"train lab quantiles info written to {lab_quantiles_path}\n")

        # initialise
        tokens = dict()
        times = dict()
        times_rel = dict()
        values = dict()
        quants = dict()
        targets = dict()

        # populate with entries
        for i in tqdm.tqdm(groups.groups):
            temp = groups.get_group(i).sort_values(by="CHARTTIME")
            assert not temp.empty, f"Empty labs for hadm:{i}. There should be {get_from_adm(i, 'NUMLABS<2D')}"
            temp['QUANT'] = temp.apply(lambda x: apply_quantile_fct(x, lab_quantiles_train, 'VALUE_SCALED'), axis=1)

            tokens[i] = np.fromiter(
                map(map2token, temp['ITEMID']),
                dtype=np.int32
            )
            times[i] = np.fromiter(
                map(ts_to_posix, temp['CHARTTIME']),
                dtype=np.int64
            )

            admittime = get_from_adm(i, 'ADMITTIME')
            times_rel[i] = times[i] - ts_to_posix(admittime)

            values[i] = np.fromiter(
                temp['VALUE_SCALED'],
                dtype=np.float64
            )
            quants[i] = np.fromiter(
                temp['QUANT'],
                dtype=np.int32
            )

            # NOTE: can refactor target extraction easily to derive from augmented_admissions.csv
            los = pd.Timedelta(get_from_adm(i, 'LOS'))
            los = np.round(los.total_seconds()/(24*60*60) - 2, decimals=5)

            targets[i] = {
                'DEATH>2.5D': get_from_adm(i, 'DEATH>2.5D'),
                'DEATH<=3D': get_from_adm(i, 'DEATH<=3D'),
                'DEATH>3D': get_from_adm(i, 'DEATH>3D'),
                'DEATH>7D': get_from_adm(i, 'DEATH>7D'),
                'DEATH<=7D': get_from_adm(i, 'DEATH<=7D'),
                'LOS': los
            }

        # write out charts to pickle
        save_path = os.path.join(args.save_root, f'{subset}_data.pkl')

        with open(save_path, 'wb') as f:
            pickle.dump({f'{subset}_tokens': tokens,
                         f'{subset}_values': values,
                         f'{subset}_quants': quants,
                         f'{subset}_times_rel': times_rel
                         },
                        f)
        del tokens, times, times_rel, groups

        # write out targets to pickle
        save_path = os.path.join(args.save_root, f'{subset}_targets.pkl')

        with open(save_path, 'wb') as f:
            pickle.dump({f'{subset}_targets': targets}, f)
        del targets

        print(f'{subset} set data processed!')

    with open(os.path.join(args.save_root, 'mappings.pkl'), 'wb') as f:
        pickle.dump({'itemid2token': itemid2token,
                     'token2itemid': token2itemid,
                     'token2trcount': token2trcount,
                     'qname2qtoken': qname2qtoken,
                     'qtoken2qname': qtoken2qname},
                    f)


def preprocess_labs_for_1D(args):
    print('*' * 17, 'preprocessing labs with the following settings:', sep='\n')
    pprint(vars(args), indent=2)
    print('*' * 17)

    # paths

    admissions_path = os.path.join(args.data_root, "augmented_admissions.csv")
    scaled_labevents_path = os.path.join(args.data_root, "labevents_scaled.csv")
    lab_quantiles_path = os.path.join(args.save_root, "lab_quantiles.csv")
    d_labitems_path = os.path.join(args.mimic_root, "D_LABITEMS.csv")

    # read labevents and summarise

    labevents = (pd.read_csv(scaled_labevents_path,
                             index_col='ROW_ID',
                             parse_dates=['CHARTTIME', 'ADMITTIME'])
                 .dropna(subset=['HADM_ID'])
                 .astype({'HADM_ID': 'int', 'VALUEUOM': 'str'})
                 )

    adm = pd.read_csv(admissions_path,
                      index_col='HADM_ID',
                      parse_dates=['ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'EDREGTIME', 'EDOUTTIME'])

    train_indices = adm[adm.PARTITION == 'train'].index.to_numpy()
    val_indices = adm[adm.PARTITION == 'val'].index.to_numpy()
    test_indices = adm[adm.PARTITION == 'test'].index.to_numpy()

    with open(os.path.join(args.data_root, 'mappings.pkl'), 'rb') as f:
        itemid2token = pickle.load(f)['itemid2token']

    def map2token(itemid):
        return itemid2token[int(itemid)]

    def get_from_adm(hadm_id, target):
        return adm.loc[hadm_id, target]

    labevents2 = labevents[labevents.CHARTTIME <= labevents.ADMITTIME + pd.Timedelta(days=2)]

    for subset in ['train', 'val', 'test']:
        print(f'Processing {subset} set data...')

        # grouper for labs
        groups = (labevents2.query(f'HADM_ID.isin(@{subset}_indices)')
                  .groupby(by='HADM_ID')
                  )

        # train token counts and quantile calculation
        if (subset == 'train') & bool(args.write_quantiles_summary):
            print("calculating train lab value quants...")
            lab_quantiles_train = groups.obj.groupby('ITEMID').VALUE_SCALED.quantile(args.quantiles)
            print("train lab value quants calculated!\n")
            lab_quantiles_train.to_csv(lab_quantiles_path)
            print(f"train lab quantiles info written to {lab_quantiles_path}\n")
        elif not bool(args.write_quantiles_summary):
            lab_quantiles_train = pd.read_csv(lab_quantiles_path, index_col=[0, 1], squeeze=True)

        num_tokens = len(itemid2token)

        # initialise
        values_mean = dict()
        values_latest = dict()
        values_count = dict()
        quants_mean = dict()

        # populate with entries
        for i in tqdm.tqdm(groups.groups):
            temp = groups.get_group(i).sort_values(by="CHARTTIME")
            temp['TOKEN'] = temp['ITEMID'].apply(map2token)

            temp_mean = temp.groupby('TOKEN')[['VALUE_SCALED', 'ITEMID']].mean()
            temp_mean['QUANT'] = temp_mean.apply(lambda x: apply_quantile_fct(x, lab_quantiles_train, 'VALUE_SCALED'),
                                                 axis=1)
            temp_mean['VALUE_SCALED'] = temp_mean['VALUE_SCALED'].fillna(args.sentinel_cat) \
                if args.sentinel_cat is not None else temp_mean
            temp_latest = (temp.groupby('TOKEN').tail(1)[['TOKEN', 'VALUE_SCALED']]
                           .set_index('TOKEN').sort_index()['VALUE_SCALED'])
            temp_latest = temp_latest.fillna(args.sentinel_cat) if args.sentinel_cat is not None else temp_latest
            temp_count = (temp.groupby('TOKEN').count()['ITEMID'])
            temp_count = temp_count.fillna(args.sentinel_cat) if args.sentinel_cat is not None else temp_count

            def make_bov_from_(df_col, sentinel_value):
                z = np.full(num_tokens, sentinel_value)
                idx = df_col.index.to_numpy()
                val = df_col.to_numpy()
                z[idx] = val
                return z

            values_mean[i] = make_bov_from_(temp_mean['VALUE_SCALED'], args.pad_mean)
            values_latest[i] = make_bov_from_(temp_latest, args.pad_latest)
            values_count[i] = make_bov_from_(temp_count, args.pad_count)
            quants_mean[i] = make_bov_from_(temp_mean['QUANT'], args.pad_quant)

        # write out labs to pickle
        save_path = os.path.join(args.save_root, f'{subset}_data1D.pkl')

        with open(save_path, 'wb') as f:
            pickle.dump({f'{subset}_values_mean': values_mean,
                         f'{subset}_values_latest': values_latest,
                         f'{subset}_values_count': values_count,
                         f'{subset}_quants_mean': quants_mean},
                        f)


if __name__ == "__main__":
    arguments = PreprocessingArguments().parse()

    if arguments.preprocess_for == '1D':
        assert arguments.data_root is not None, "To preprocess for 1D we need to have done 1.5D preprocessing!"
        preprocess_labs_for_1D(arguments)
    elif arguments.preprocess_for == '1.5D':
        preprocess_labs_for_1p5D(arguments)
    else:
        raise Exception("Must specify which data format we are preprocessing for!")
