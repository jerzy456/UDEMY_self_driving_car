def RNN_batch_generator(image_paths, steering_ang, batch_size, timesteps, istraining):
    while True:
        batch_idx = []
        # choose bacth indexes
        random_indexes = sample([j for j in range(timesteps, len(image_paths) - 1)], batch_size)
        # append "timesteps" frames before choosen index
        for j in range(batch_size):
            batch_idx.append(range(random_indexes[j], random_indexes[j] - timesteps, -1))
        batch_idx = np.asarray(batch_idx)

        batch_img = [[], []]
        batch_steering = [[], []]
        for i in range(batch_idx.shape[1]):
            for j in range(batch_idx.shape[0]):
                if istraining:
                    im, steering = random_augment(image_paths[batch_idx[j][i]], steering_ang[batch_idx[j][i]])
                else:
                    im = mpimg.imread(image_paths[batch_idx[j][i]])
                    steering = steering_ang[batch_idx[j][i]]
                im = img_preprocess(im)
                batch_img[i].append(im)
                batch_steering[i].append(steering)
        yield (np.asarray(batch_img), np.asarray(batch_steering))