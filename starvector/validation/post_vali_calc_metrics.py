import os
import json
import pandas as pd
from starvector.metrics.metrics import SVGMetrics
from starvector.data.util import rasterize_svg
from omegaconf import OmegaConf
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader

class PostValiCalcMetrics:
    def __init__(self, config):
        print(f"[DEBUG] Config: {config}")
        self.task = config.model.task
        self.date_time = config.run.date_time
        print(f"[DEBUG] Date time type: {type(self.date_time)}")
        print(f"[DEBUG] Date time: {self.date_time}")
        self.model_name = config.model.name
        self.out_dir = config.run.out_dir + '/' + config.model.generation_engine + '_' + config.model.name + '_' + config.dataset.dataset_name + '_' + self.date_time
        # Check if the out_dir exists
        if not os.path.exists(self.out_dir):
            raise FileNotFoundError(f"The out_dir {self.out_dir} does not exist")
        print(f"Out dir: {self.out_dir}")

        metrics_config_path = f"configs/metrics/{self.task}.yaml"
        default_metrics_config = OmegaConf.load(metrics_config_path)
        self.metrics = SVGMetrics(default_metrics_config['metrics'])
        self.results = {}
        self.config = config
        self.get_dataloader()
    
    def get_dataloader(self):
        data = load_dataset(self.config.dataset.dataset_name, self.config.dataset.config_name, split=self.config.dataset.split)

        if self.config.dataset.num_samples != -1:
            data = data.select(range(self.config.dataset.num_samples))

        self.dataloader = DataLoader(data, batch_size=1, shuffle=False)
    
    def post_vali_calc_metrics(self):
        """Main Post-Validation Calculation Metrics"""
        for i, sample in enumerate(tqdm(self.dataloader, desc="Calculating metrics")):
            if self.task == "text2svg":
                # caption_blip2, caption_cogvlm, caption_llava
                # add 'caption' to batch
                # print("[DEBUG] Caption key:", self.config.dataset.caption_key)
                sample['caption'] = sample[self.config.dataset.caption_key]

            result = self.load_processed_sample(sample)
            self.results[result['sample_id']] = result

        self.calculate_and_save_metrics()
            
    def load_processed_sample(self, sample):
        """
        Load processed sample from the out_dir
        Args:
            sample: sample from the dataloader
        Returns:
            result: processed sample
            {
                "caption": str,
                "gt_svg": str,
                "non_compiling": bool,
                "post_processed": bool,
                "sample_id": str,
                "svg": str,
                "raw_svg": str,
                "gen_im": PIL.Image.Image,
                "gt_im": PIL.Image.Image,
            }
        """
        sample_id = str(sample['Filename'][0]).split('.')[0]
        sample_dir = os.path.join(self.out_dir, sample_id)
        with open(os.path.join(sample_dir, 'metadata.json'), 'r') as f:
            result = json.load(f)
        svg_raster = rasterize_svg(result['svg'], resolution=512, dpi=100, scale=1)
        gt_svg_raster = rasterize_svg(result['gt_svg'], resolution=512, dpi=100, scale=1)
        result['gen_im'] = svg_raster
        result['gt_im'] = gt_svg_raster
        return result

    def calculate_and_save_metrics(self):
        processed_results = self.preprocess_results()
        avg_results, all_results = self.metrics.calculate_metrics(processed_results)
        out_path_results = os.path.join(self.out_dir, 'results')
        os.makedirs(out_path_results, exist_ok=True)
        with open(os.path.join(out_path_results, 'results_avg.json'), 'w') as f:
            json.dump(avg_results, f, indent=4, sort_keys=True)
        df = pd.DataFrame.from_dict(all_results, orient='index')
        df.to_csv(os.path.join(out_path_results, 'all_results.csv'))

        self.create_comparison_plots_with_metrics(all_results)

    def preprocess_results(self):
        processed_results = {
            'gen_svg': [],
            'gt_svg': [],
            'gen_im': [],
            'gt_im': [],
            'caption': [],
            'json': []
        }

        for _, result_dict in self.results.items():
            processed_results['gen_svg'].append(result_dict['svg'])
            processed_results['gt_svg'].append(result_dict['gt_svg'])
            processed_results['gen_im'].append(result_dict['gen_im'])
            processed_results['gt_im'].append(result_dict['gt_im'])
            processed_results['caption'].append(result_dict['caption'])
            processed_results['json'].append(result_dict)

        return processed_results
    
    def create_comparison_plot(self, sample_id, gt_raster, gen_raster, metrics, output_path):
        """
        Creates and saves a comparison plot showing the ground truth and generated SVG images, along with computed metrics.
        
        Args:
            sample_id (str): Identifier for the sample.
            gt_raster (PIL.Image.Image): Rasterized ground truth SVG image.
            gen_raster (PIL.Image.Image): Rasterized generated SVG image.
            metrics (dict): Dictionary of metric names and their values.
            output_path (str): File path where the plot is saved.
            
        Returns:
            PIL.Image.Image: The generated comparison plot image.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from io import BytesIO
        from PIL import Image

        # Create figure with two subplots: one for metrics text, one for the images
        fig, (ax_metrics, ax_images) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 4]})
        fig.suptitle(f'Generation Results for {sample_id}', fontsize=16)

        # Build text for metrics
        if metrics:
            metrics_text = "Metrics:\n"
            for key, val in metrics.items():
                if isinstance(val, list) and val:
                    metrics_text += f"{key}: {val[-1]:.4f}\n"
                elif isinstance(val, (int, float)):
                    metrics_text += f"{key}: {val:.4f}\n"
                else:
                    metrics_text += f"{key}: {val}\n"
        else:
            metrics_text = "No metrics available."
        
        # Add metrics text in the upper subplot
        ax_metrics.text(0.5, 0.5, metrics_text, fontfamily='monospace',
                        horizontalalignment='center', verticalalignment='center')
        ax_metrics.axis('off')

        # Set title and prepare the images subplot
        ax_images.set_title('Ground Truth (left) vs Generated (right)')
        gt_array = np.array(gt_raster)
        gen_array = np.array(gen_raster)
        combined = np.hstack((gt_array, gen_array))
        ax_images.imshow(combined)
        ax_images.axis('off')

        # Save figure to buffer and file path
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        plt.savefig(output_path, format='png', bbox_inches='tight', dpi=300)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)
    
    def create_comparison_plots_with_metrics(self, all_metrics):
        """
        Create and save comparison plots with metrics for all samples based on computed metrics.
        """
        for sample_id, metrics in tqdm(all_metrics.items(), desc="Creating comparison plots"):
            if sample_id not in self.results:
                continue  # Skip if the sample does not exist in the results
            
            result = self.results[sample_id]
            sample_dir = os.path.join(self.out_dir, sample_id)
            
            # Retrieve the already rasterized images from the result
            gt_raster = result.get('gt_im')
            gen_raster = result.get('gen_im')
            if gt_raster is None or gen_raster is None:
                continue
            
            # Define the output path for the comparison plot image
            output_path = os.path.join(sample_dir, f"{sample_id}_comparison.png")
            comp_img = self.create_comparison_plot(sample_id, gt_raster, gen_raster, metrics, output_path)
            
            # Save the generated plot image in the result for later use
            result['comparison_image'] = comp_img

            self.results[sample_id] = result

def check_gpu_memory_and_usage():
    import torch
    import psutil
    import os
    import time
    import gc
    import matplotlib.pyplot as plt
    import numpy as np

    # Get GPU memory usage
    def get_gpu_memory_usage():
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 3)  # in GB
        return 0

    # Get CPU memory usage
    def get_cpu_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 3)  # in GB

    print(f"[DEBUG] GPU memory usage: {get_gpu_memory_usage()} GB")
    print(f"[DEBUG] CPU memory usage: {get_cpu_memory_usage()} GB")

if __name__ == "__main__":
    check_gpu_memory_and_usage()
    cli_conf = OmegaConf.from_cli()
    if 'config' not in cli_conf:
        raise ValueError("No config file provided. Please provide a config file using 'config=path/to/config.yaml'")
    
    config_path = cli_conf.pop('config')
    config = OmegaConf.load(config_path)
    config = OmegaConf.merge(config, cli_conf)

    post_vali_calculator = PostValiCalcMetrics(config)
    post_vali_calculator.post_vali_calc_metrics()
