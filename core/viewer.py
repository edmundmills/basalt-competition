

class TrajectoryViewer:
    def dataset_recent_frames(self, dataset, number_of_steps):
        total_steps = len(dataset.step_lookup)
        steps = min(number_of_steps, total_steps)
        frame_skip = 2
        frames = int(round(total_steps / (frame_skip + 1)))
        step_rate = 20  # steps / second
        frame_rate = int(round(step_rate / (frame_skip + 1)))
        step_indices = [min(total_steps - steps + frame * (frame_skip + 1),
                            total_steps - 1)
                        for frame in range(frames)]
        indices = [dataset.step_lookup[step_index] for step_index in step_indices]
        images = [dataset.trajectories[trajectory_idx].get_pov(step_idx)
                  for trajectory_idx, step_idx in indices]
        images = [(image.numpy()).astype(np.uint8)
                  for image in images]
        images = np.stack(images, 0)
        return images, frame_rate
