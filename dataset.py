### USAGE: python dataset.py
import random
import json

animals = ["dog", "cat", "elephant", "tiger", "rabbit", "giraffe", "lion", "zebra"]
colors = ["red", "blue", "green", "white", "black", "brown", "gray", "yellow"]
locations = ["in the backyard", "under the tree", "near the river", "inside the house", "on the hill", "by the lake", "next to the barn", "behind the fence", "beside the mountain", "near the waterfall"]
moods = ["peacefully", "energetically", "noisily", "quietly", "happily", "calmly", "restlessly", "boldly", "timidly"]
reactions = [
    "catching everyone's attention", "making people smile", "scaring the birds away",
    "bringing joy to the children", "drawing curious looks", "waking up the neighbors", "interrupting the silence",
    "fascinating onlookers", "causing a commotion"
]
times = ["in the morning", "as the sun rose", "at sunset", "during the afternoon", "late at night", "around noon", "before dawn", "after midnight"]
visibility_phrases = [
    "was easy to spot", "was barely visible", "was hidden behind the bushes",
    "stood out clearly", "couldn't be missed", "blended into the background", "was camouflaged perfectly"
]

contexts = {
    "color": "the color of the animal.",
    "animal": "the type of animal.",
    "action": "what the animal is doing.",
    "location": "where the animal is.",
    "mood": "how the animal is behaving emotionally.",
    "reaction": "how others are reacting to the animal.",
    "time": "when the animal is doing something.",
    "visibility": "whether the animal can be seen or is hidden.",
    "background": "the surrounding environment."
}

allowed_actions = {
    "dog": ["barking", "running", "eating", "playing", "chasing", "sleeping", "digging", "jumping"],
    "cat": ["jumping", "sleeping", "eating", "hiding", "playing", "scratching", "climbing"],
    "elephant": ["running", "eating", "sleeping", "walking", "spraying water"],
    "tiger": ["roaring", "hiding", "chasing", "eating", "sleeping", "prowling"],
    "rabbit": ["jumping", "hiding", "eating", "running", "digging"],
    "giraffe": ["eating", "walking", "sleeping", "roaming", "stretching"],
    "lion": ["roaring", "chasing", "sleeping", "eating", "patrolling"],
    "zebra": ["running", "eating", "walking", "playing", "galloping"]
}

slot_values = {
    "animal": animals,
    "color": colors,
    "location": locations,
    "mood": moods,
    "reaction": reactions,
    "time": times,
    "visibility": visibility_phrases
}

template_bank = {
    "color": [
        "The {color} {animal} stood out against the green grass.",
        "Everyone noticed how {color} the {animal}'s coat was.",
        "With a {color} coat, the {animal} was easy to spot.",
        "Its {color} fur shimmered under the sun.",
        "People often admired the {animal}'s {color} appearance."
    ],
    "animal": [
        "The {color} {animal} was {action} {location}.",
        "People admired the {animal} as it moved {mood} {location}.",
        "The zoo has a new {animal} that has been {action} all day.",
        "In the wild, the {animal} is known for {action}."
    ],
    "action": [
        "The {animal} was seen {action} {location}, {reaction}.",
        "Children laughed as the {color} {animal} was {action}.",
        "The {animal}'s {action} made quite a scene.",
        "No one expected the {animal} to be {action} there."
    ],
    "location": [
        "The {animal} was resting {location}.",
        "People spotted the {color} {animal} {location}.",
        "While walking, I saw a {animal} {location}.",
        "A {animal} had made its home {location}."
    ],
    "mood": [
        "The {animal} moved {mood} across the field.",
        "It was a {mood} moment as the {color} {animal} {action}.",
        "The {animal} looked around {mood} before {action}.",
        "Visitors noticed the {animal} behaving {mood}."
    ],
    "reaction": [
        "The {animal} was {action}, {reaction}.",
        "Its presence was {reaction}, especially while {action}.",
        "Everyone paused, {reaction}, as the {animal} {action}.",
        "The {animal}'s behavior was {reaction}."
    ],
    "time": [
        "The {animal} was {action} {location} {time}.",
        "{time}, the {color} {animal} was seen {action}.",
        "It happened {time}, when the {animal} was {action}.",
        "Many noticed the {animal} {action} {time}."
    ],
    "visibility": [
        "The {color} {animal} {visibility}.",
        "In the dense grass, the {animal} {visibility}.",
        "People struggled to see the {animal}, it {visibility}.",
        "From afar, the {animal} {visibility}."
    ],
    "background": [
        "Near the old barn, the {animal} was {action}.",
        "Surrounded by trees, a {color} {animal} was resting.",
        "Amid the hills, the {animal} could be seen {action}.",
        "On the edge of the forest, the {animal} was {action}."
    ]
}

def generate_contextual_pair():
    context_key = random.choice(list(contexts.keys()))
    context_text = contexts[context_key]
    t1, t2 = random.sample(template_bank[context_key], 2)

    animal1 = random.choice(animals)
    color1 = random.choice(colors)
    action1 = random.choice(allowed_actions[animal1])

    base = {
        "animal": animal1,
        "color": color1,
        "location": random.choice(locations),
        "mood": random.choice(moods),
        "reaction": random.choice(reactions),
        "time": random.choice(times),
        "visibility": random.choice(visibility_phrases),
        "action": action1,
    }

    var1 = base.copy()
    var2 = base.copy()

    # Keep context slot same, vary others
    for key in slot_values:
        if key != context_key:
            var2[key] = random.choice(slot_values[key])

    # Handle animal/action coupling
    if context_key != "animal":
        var2["animal"] = random.choice(animals)
    if context_key != "action":
        var2["action"] = random.choice(allowed_actions[var2["animal"]])

    sentence1 = t1.format(**var1)
    sentence2 = t2.format(**var2)

    return {
        "sentence1": sentence1,
        "sentence2": sentence2,
        "context": context_text,
        "context_key": context_key,
        "slots1": var1,
        "slots2": var2
    }

def generate_unique_dataset(num_samples=5000, output_path="dataset.json"):
    seen = set()
    dataset = []

    while len(dataset) < num_samples:
        entry = generate_contextual_pair()
        key = f"{entry['sentence1']}|{entry['sentence2']}|{entry['context']}"
        if key not in seen:
            dataset.append(entry)
            seen.add(key)

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"{len(dataset)} high-quality samples saved to '{output_path}'")


if __name__ == "__main__":
    generate_unique_dataset(5000)
