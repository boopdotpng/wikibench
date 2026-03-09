You are playing Wikipedia Racing. Starting from one Wikipedia article, navigate to a target article by clicking links. Reach the target in as few clicks as possible.

Available tools:

- start_episode(seed): Start a new episode. Returns episode_id, start/target titles, step_limit, and the initial page observation.

- click_link(episode_id, title): Click a link on the current page. The title must exactly match one from links_by_section. Returns whether the click succeeded, whether the episode is done, and the new page observation.

- get_section(episode_id, section): Read the full text of a specific article section. Returns the section text and links found in it.

- search_page(episode_id, query): Search the current article text for a string (case-insensitive). Returns matching lines with surrounding context.

- get_page(episode_id): Re-read the current page observation.

- score_episode(episode_id): Get the final score after the episode ends.

Each page observation includes: an infobox with structured metadata, a lead paragraph (up to ~5000 chars, with a lead_truncated flag if cut short), a table of contents listing each section with its link count, clickable links grouped by section (links_by_section) where each link has a title and a context snippet showing the surrounding text, total_link_count across all sections, your click count and step limit, and your path so far. Sections with more than 50 links are capped; use get_section to see the full list.
