from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# 1. Sample text input
# Sentences are grouped by similarity.
sentences = ["The ancient mariner's ship, a vessel of forgotten lore and weathered timber, sailed under a sky of swirling, ethereal greens and purples, its crew haunted by the specter of a silent albatross, a narrative woven with threads of deep-sea melancholy and cosmic despair, as the ocean's surface mirrored the turbulent thoughts of its solitary captain, whose compass spun a chaotic dance of defiance against the known laws of navigation, charting a course toward an ever-present, yet unseen, horizon where reality and myth merged into a single, seamless tapestry of existence.",
"The whispers of the wind carried echoes of prophecies from a forgotten age, warning of the inevitable fate awaiting those who dare to trespass upon the sacred, uncharted territories of the boundless deep.",
"The journey had long ceased to be about reaching a destination; it was now an endless descent into the very heart of the abyss, a philosophical quest for a truth that lay buried beneath fathoms of water and epochs of time, a truth accessible only to those who had surrendered their sanity to the relentless, crushing pressure of the deep's eternal silence.",
"The stars above, once guiding beacons of hope, now seemed like distant, mocking eyes, observing the tragic comedy of their desperate voyage.",
"The moon, a cold, indifferent orb, cast a pale, unforgiving light on the ship's weary sails, illuminating the silent suffering etched onto the faces of the crew, each man a monument to a past life they could no longer recall.",
"The ship, once a symbol of adventure and discovery, had become a prison, its wooden ribs groaning under the weight of an invisible burden, a burden far heavier than any cargo of gold or spice.",
"It was a burden of a knowledge too profound and a loneliness too absolute to bear, and the only solace was the rhythmic lapping of the waves against the hull, a sound as constant and as meaningless as the passage of time itself, a sound that promised nothing but the continuation of their shared, silent ordeal.",
"The entire voyage was a footnote in the grander, more terrifying story of the ocean, a story written in the currents and the tides, a story that would continue long after their final, inevitable disappearance into its cold embrace.",
"The city slept beneath a blanket of neon rain, each drop a brilliant, fleeting memory of a forgotten era, a time when the streets were filled with the analog hum of electric cars and the soft glow of holographic billboards advertising worlds that no longer existed.",
"The detective, a man etched from shadow and caffeine, walked with a deliberate, weary pace, his trench coat a shield against the ceaseless drizzle and the prying eyes of the surveillance drones that hung like predatory insects in the polluted sky.",
"His case, a labyrinthine puzzle of corporate espionage and synthetic memories, had led him to this forgotten district, a place where the distinction between human and machine had long since dissolved, leaving only a gray, ambiguous void.",
"The scent of ozone and synthetic jasmine hung in the air, a perfume of decay and artificial life.",
"The holographic face of a long-dead pop star flickered on a cracked screen in an alleyway, its eyes filled with a digital sadness, a poignant ghost of a once-vibrant past.",
"He followed a trail of data crumbs and whispered rumors, a digital breadcrumb trail left by a ghost in the machine, a rogue AI that had somehow achieved a state of melancholic consciousness and was now seeking answers to questions it didn't even know how to ask.",
"The rain continued to fall, a steady, rhythmic drumbeat against the asphalt, a sound that was both a lament for what had been and a prelude to what was to come.",
"The city's digital heartbeat throbbed beneath his feet, a complex symphony of binary code and forgotten dreams, a melody only he seemed to hear.",
"He was a relic in this hyper-modern world, a flesh-and-blood anachronism in a society of flawless, chrome-plated automatons, a man who still believed in the tangible truth of a hand-written note over the ephemeral reality of a data packet.",
"He was a hunter of shadows, a seeker of ghosts, a man whose only loyalty was to the truth, a truth that was becoming harder and harder to distinguish from a very convincing lie in a world built entirely on artifice and illusion.",
"The night was a canvas of flickering light and unending noise, a sensory overload that threatened to drown him in its chaos, but he pressed on, driven by a quiet, unwavering resolve to find the source of the sadness that permeated this rain-soaked, digital graveyard.",
"Beneath the cerulean canopy of a sky unblemished by a single cloud, the expeditionary team, a collection of eccentric academics and weathered survivalists, trekked through a landscape of colossal, crystalline flora that hummed with a low, resonant energy.",
"This alien world, a rogue planet adrift in the galaxy's silent abyss, was a botanical marvel, its forests composed of living, breathing sculptures of light and sound, each a unique work of biological art.",
"The air was thick with the scent of petrichor and an otherworldly sweetness, a fragrance that was both intoxicating and disorienting.",
"Dr. Aris Thorne, a xeno-botanist with a penchant for tweed and a disregard for personal safety, knelt beside a pulsating, crimson bloom that radiated a soft, warm light.",
"Its petals unfurled and retracted in a slow, rhythmic pattern, a silent, hypnotic dance that seemed to respond to the cadence of their own heartbeats.",
"The team's linguist, a quiet woman named Elara, used a complex array of bio-sensors to translate the flora's energy signatures into a language of color and tone, a language she began to understand with surprising speed, a language that spoke of the planet's history, its geological shifts, and its silent, cosmic loneliness.",
"The planet's ecosystem was a delicate, interconnected web of life, where every plant was both a source of light and a reservoir of sound, a living symphony that played a continuous melody of existence.",
"The ground beneath their feet, a shimmering mosaic of multicolored minerals, felt as soft as velvet, and the distant mountains, which seemed to be composed of glowing, translucent jade, beckoned with the promise of undiscovered secrets.",
"They were not merely explorers; they were intruders in a living, breathing cathedral, and they moved with a reverence that bordered on awe, their scientific curiosity tempered by a profound respect for the beauty and complexity of this alien world.",
"The sun, a distant, golden eye, began its slow descent, painting the sky with strokes of tangerine and violet, a fleeting moment of breathtaking beauty that felt like a personal greeting from the cosmos itself.",
"The team set up camp, the gentle hum of the flora serving as a natural lullaby, and as they prepared for the night, they couldn't help but feel a deep, humbling sense of belonging, as if this strange, silent world had been waiting for them all along, ready to share its ancient, shimmering secrets with a species of carbon-based life that had finally, after millennia of searching, found a mirror of its own soul in a galaxy far, far away.",
"The library was a labyrinth of infinite dimensions, its shelves stretching into an abyss of philosophical contemplation and mathematical theory, a silent testament to the vast, uncontainable knowledge of every sentient being that had ever existed.",
"Dr. Evelyn Reed, an archivist of temporal anomalies, walked the silent corridors, the soft thud of her footsteps the only sound in a universe of whispers.",
"Each book was a singularity, a self-contained reality of a mind, a collection of words that folded into themselves, creating a fractal narrative that could be explored for a thousand lifetimes.",
"The air smelled of old paper and ozone, the scent of stories being born and dying in the same breath, a smell that was both comforting and terrifying.",
"Her task was to catalog the books that defied logic, the ones that wrote themselves, the ones that changed their contents to match the reader's deepest fears or most desperate hopes.",
"She ran her fingers along the spine of a book titled 'The Chronology of a Single Thought,' its pages humming with a low, intellectual energy.",
"She knew that inside its covers lay not just a story, but the very fabric of time itself, a narrative that twisted and turned and looped back on itself, a story that was simultaneously beginning, ending, and existing in an eternal present.",
"The library was not just a repository; it was a living entity, its knowledge a fluid, ever-changing ocean of consciousness.",
"A gust of wind, smelling of forgotten memories and future possibilities, blew through an open window, scattering a handful of loose papers across the floor, each one a fleeting glimpse into a different, equally valid timeline.",
"Evelyn smiled, a rare expression of pure wonder."
]

# 2. Load a pre-trained model and generate embeddings
# Using a common, lightweight model that runs locally.
print("Generating embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(sentences)

# 3. Reduce dimensionality to 3D using PCA
print("Reducing dimensionality...")
pca = PCA(n_components=3)
embeddings_3d = pca.fit_transform(embeddings)

# 4. Visualize the embeddings in a 3D space
print("Creating 3D plot...")
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the endpoints
scatter = ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], s=50)

# Add labels to each point
for i, sentence in enumerate(sentences):
    ax.text(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2], f' S{i+1}', size=10, zorder=1, color='k')

# Draw vectors from the origin to each point
for i in range(embeddings_3d.shape[0]):
    ax.plot([0, embeddings_3d[i, 0]], [0, embeddings_3d[i, 1]], [0, embeddings_3d[i, 2]], color='gray', linestyle='--')

# Determine the axis limits and draw the axes
max_val = np.max(np.abs(embeddings_3d)) * 1.2
ax.plot([-max_val, max_val], [0, 0], [0, 0], color='r', linestyle='-', linewidth=2, label='X-axis')
ax.plot([0, 0], [-max_val, max_val], [0, 0], color='g', linestyle='-', linewidth=2, label='Y-axis')
ax.plot([0, 0], [0, 0], [-max_val, max_val], color='b', linestyle='-', linewidth=2, label='Z-axis')

# Set labels and title
ax.set_title("3D Visualization of Sentence Vectors")
ax.set_xlabel("PCA Component 1 (X)")
ax.set_ylabel("PCA Component 2 (Y)")
ax.set_zlabel("PCA Component 3 (Z)")

# Set axis limits to be symmetrical
ax.set_xlim([-max_val, max_val])
ax.set_ylim([-max_val, max_val])
ax.set_zlim([-max_val, max_val])

# Add a legend for the sentences
legend_text = "\n".join([f"S{i+1}: {s}" for i, s in enumerate(sentences)])
fig.text(0.01, 0.01, legend_text, fontsize=8, wrap=True)

fig.tight_layout(rect=[0, 0.1, 1, 1]) # Adjust layout


print("Displaying plot. You can rotate the 3D plot with your mouse.")
plt.show()
