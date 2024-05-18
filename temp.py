# Blit the scaled image onto the screen
grouped_images = dict()
# find images of same parent
for i in range(len(screen_ims)):
    if screen_parents[i] in grouped_images:
        grouped_images[screen_parents[i]].append((screen_ims[i], screen_scores[i], screen_actions[i], screen_parents[i], screen_self[i]))
    else:
        grouped_images[screen_parents[i]] = [(screen_ims[i], screen_scores[i], screen_actions[i], screen_parents[i], screen_self[i]), ]

number_images_drawn = 0
for group_id, group in enumerate(grouped_images):
    if split_depths:
        #line_color = (*boardState._get_unique_color((depth+depth_offset+1)/unique_parents * 255), line_opacity)
        line_color = (*boardState._get_unique_color((group_id)/len(grouped_images) * 255), line_opacity)
        pass
    else:
        #line_color = (*boardState._get_unique_color((groups_drawn+1)/unique_parents * 255), line_opacity)
        line_color = (*boardState._get_unique_color((number_images_drawn+1)/len(screen_ims) * 255), line_opacity)
        groups_drawn += 1

    if not disable_printing:
        print("Depth: ", depth, ", Images: ", len(grouped_images[group]))
    for i, data in enumerate(grouped_images[group]):
        im, score, actions, parent, slf = data
        score_surface = my_font.render("Score: " + str(round(score,2)), True, snake_background_color)
        depth_surface = my_font.render(f'Depth: {depth}', True, snake_background_color)
        parent_surface = my_font.render(f'Parent: {round(parent,7)}', True, snake_background_color) 
        id_surface = my_font.render(f'ID: {round(slf,7)}', True, snake_background_color) 
        action_surfaces = []
        for snake_id in actions:
            _s, _ = boardState.get_snake(snake_id)
            color_name = get_colour_name(boardState._get_unique_color(_s["m"]))
            action_txt = color_name + ": " + actions[snake_id].name
            _surface = my_font.render(action_txt, True, snake_background_color)
            action_surfaces.append(_surface)

        # centering + image offset between images
        if split_depths:
            x_diff = (large_width/2 - len(grouped_images[group]) * (image_width/2 + image_x_separation/2)) + i*image_width + (i+0.5)*image_x_separation
        else:
            x_diff = (large_width/2 - len(screen_ims) * (image_width/2 + image_x_separation/2)) + number_images_drawn*image_width + (number_images_drawn+0.5)*image_x_separation
        
        number_images_drawn += 1
        # depth offset + image offset at depth
        im_y_diff = (depth + depth_offset)*image_height + (depth + depth_offset)*image_y_separation+starting_y
        action_y_diffs = []
        for j in range(len(action_surfaces)):
            action_y_diffs.append(im_y_diff-text_y_separation*(j+1))
        score_y_diff = action_y_diffs[-1] - text_y_separation if len(action_y_diffs) > 0 else im_y_diff - text_y_separation
        depth_y_diff = score_y_diff-text_y_separation
        parent_y_diff = depth_y_diff-text_y_separation
        id_y_diff = parent_y_diff-text_y_separation

        image_location[slf] = (x_diff + image_width/2, im_y_diff + image_height/2)

        # Need to draw images above lines
        image_buffer.append((im, (x_diff, im_y_diff)))
        image_buffer.append((score_surface, (x_diff, score_y_diff)))
        for j in range(len(action_surfaces)):
            image_buffer.append((action_surfaces[j], (x_diff, action_y_diffs[j])))
        image_buffer.append((depth_surface, (x_diff, depth_y_diff)))
        image_buffer.append((parent_surface, (x_diff, parent_y_diff)))
        image_buffer.append((id_surface, (x_diff, id_y_diff)))

        if parent >= 0: # Dont draw a line for the top node
            start_pos = (x_diff + image_width/2, im_y_diff)
            line_buffer.append(((start_pos, image_location[parent]), line_color))

    if (split_depths): # and len(grouped_images) > 1): # this makes it so that only nodes with multiple parents get split # and (group_id+1) < len(grouped_images)): # This removes the gap between depths
        depth_offset += 1