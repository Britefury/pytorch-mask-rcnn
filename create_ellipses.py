import click

@click.command()
@click.option('--path', type=click.Path(file_okay=False, dir_okay=True))
@click.option('--n_train', type=int, default=32)
@click.option('--n_test', type=int, default=32)
@click.option('--width', type=int, default=256)
@click.option('--height', type=int, default=256)
@click.option('--seed', type=int, default=12345)
def create_ellipses(path, n_train, n_test, width, height, seed):
    import os
    import tqdm
    import pickle
    import numpy as np
    from PIL import Image
    from examples.ellipses import ellipses_dataset

    image_size = height, width

    rng = np.random.RandomState(seed)

    def generate_samples(out_path, n_samples, rng):
        hulls = []

        # Generate data
        for i in tqdm.tqdm(list(range(n_samples))):
            x, y, hulls_i = ellipses_dataset.make_sample(image_size, rng=rng)

            rgb_path = os.path.join(out_path, 'rgb_{:06d}.png'.format(i))
            labels_path = os.path.join(out_path, 'labels_{:06d}.png'.format(i))

            Image.fromarray(x).save(rgb_path)
            Image.fromarray(y.astype(np.uint32)).save(labels_path)
            hulls.append(hulls_i)

        with open(os.path.join(out_path, 'convex_hulls.pkl'), 'wb') as f_hulls:
            pickle.dump(hulls, f_hulls)

    if path is None:
        path = ellipses_dataset._get_ellipses_root_dir(exists=False)

    if n_test > 0:
        test_dir = os.path.join(path, 'test')
        os.makedirs(test_dir, exist_ok=True)
        generate_samples(test_dir, n_test, rng)

    if n_train > 0:
        train_dir = os.path.join(path, 'train')
        os.makedirs(train_dir, exist_ok=True)
        generate_samples(train_dir, n_train, rng)


if __name__ == '__main__':
    create_ellipses()