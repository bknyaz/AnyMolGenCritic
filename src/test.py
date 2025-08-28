"""
python test.py --dataset_name zinc --num_layers 3 --bf16 --dropout 0.0 --lambda_predict_prop 1.0 --randomize_order --start_random --scaling_type std --special_init --nhead 16 \
 --swiglu --expand_scale 2.0 --max_len 250 --gpt --no_bias --rmsnorm --rotary --not_allow_empty_bond --guidance 1.5 --guidance_ood 1.5 --only_ood --no_test_step --best_out_of_k 5 \
  --guidance_rand --ood_values 300 1.5 0.7 --csv_file smiles.csv
"""

import argparse
from rdkit import Chem
import torch
import lightning as pl
from train import CondGeneratorLightningModule


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    CondGeneratorLightningModule.add_args(parser)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--csv_file', type=str, default='test_samples.csv')
    parser.add_argument('--load_checkpoint_path', type=str, default='./checkpoints/exp_zinc50epoch')
    hparams = parser.parse_args()
    hparams.test = True  # make sure we are in test mode
    hparams.n_gpu = 1
    hparams.num_samples = 100  # number of samples to generate for testing
    hparams.num_samples_ood = 100
    hparams.sample_batch_size = 10  # batch size for sampling during testing

    ## Add any specifics to your dataset here in terms of what to test, max_len expected, and properties (which are binary, which are continuous)
    # Note that currently, we only allow binary or continuous properties (not categorical properties with n_cat > 2)
    hparams.graph_dit_benchmark = False
    if hparams.gflownet:
        hparams.n_properties = 4
        hparams.cat_var_index = []
    elif hparams.dataset_name in ['bbbp', 'bace', 'hiv']:
        hparams.graph_dit_benchmark = True
        hparams.n_properties = 3
        if hparams.dataset_name == 'bbbp':
            assert hparams.max_len >= 200
        if hparams.dataset_name == 'bace':
            assert hparams.max_len >= 150
        if hparams.dataset_name == 'hiv':
            assert hparams.max_len >= 150
        hparams.cat_var_index = [0]
    elif hparams.dataset_name in ['zinc', 'qm9', 'moses', 'chromophore']:
        hparams.n_properties = 3
        if hparams.dataset_name == 'zinc':
            assert hparams.max_len >= 150
        if hparams.dataset_name == 'qm9':
            assert hparams.max_len >= 50
        if hparams.dataset_name == 'chromophore':
            assert hparams.max_len >= 500
        hparams.cat_var_index = []
    else:
        raise NotImplementedError()
    hparams.cont_var_index = [i for i in range(hparams.n_properties) if i not in hparams.cat_var_index]

    print('Warning: Note that for both training and metrics, results will only be reproducible when using the same number of GPUs and num_samples/sample_batch_size')
    pl.seed_everything(hparams.seed, workers=True) # use same seed, except for the dataloaders
    model = CondGeneratorLightningModule(hparams)
    is_cpu = hparams.cpu or not torch.cuda.is_available()
    if hparams.load_checkpoint_path != "":

        # split the checkpoint into two parts: transformer layers and others and save them separately
        # to upload to github directly (but better use huggingface)
        # fp = torch.load(hparams.load_checkpoint_path + '/last.ckpt',
        #                                  map_location='cpu' if is_cpu else 'cuda',
        #                                  weights_only=False)
        # state = fp['state_dict']
        # transformer_part = {k.replace('model.', ''): v for k, v in state.items() if k.startswith('model.transformer.transformer.h.0')}
        # others_part = {k.replace('model.', ''): v for k, v in state.items() if not k.startswith('model.transformer.transformer.h.0')}
        # torch.save(transformer_part, hparams.load_checkpoint_path + '/transformer0.pt')
        # torch.save(others_part,hparams.load_checkpoint_path + '/others.pt')

        d = torch.load(hparams.load_checkpoint_path + '/transformer0.pt', weights_only=True)
        d.update(torch.load(hparams.load_checkpoint_path + '/others.pt', weights_only=True))

        model.model.load_state_dict(d, strict=True)
        print('loaded model from', hparams.load_checkpoint_path)

    if hparams.compile:
        # not tested
        model = torch.compile(model)
    pl.seed_everything(hparams.seed, workers=True) # different seed per worker
    smiles_list, valid_mols_list, unique_smiles_set, gen_molwt, gen_logp, gen_qed = model.check_samples_ood(
        num_samples=hparams.num_samples,
        multi_prop=True,
        return_results=True)
    print(f'valid: {len(valid_mols_list)}/{len(smiles_list)}, unique: {len(unique_smiles_set)}/{len(smiles_list)}')

    # dump to csv
    with open(hparams.csv_file, 'w') as f:
        f.write('smiles,molwt,logp,qed\n')
        for i in range(len(valid_mols_list)):
            smiles = Chem.MolToSmiles(valid_mols_list[i])
            print(i, smiles, gen_molwt[i].item(), gen_logp[i].item(), gen_qed[i].item())
            f.write(f'{smiles},{gen_molwt[i].item()},{gen_logp[i].item()},{gen_qed[i].item()}\n')

    print('done')