# Test case definitions for concept composition
COMPOSITION_TEST_CASES = {
    "dog_car": {
        "concept1": {
            "name": "dog",
            "prompts": [
                "a photo of a dog",
            ],
        },
        "concept2": {
            "name": "car",
            "prompts": [
                "a photo of a car",
            ],
        },
        "composition": {
            "name": "dog_and_car",
            "prompts": [
                "a photo of a dog and a car",
            ],
        },
    },
    "boy_bicycle": {
        "concept1": {
            "name": "boy",
            "prompts": [
                "a photo of a boy",
            ],
        },
        "concept2": {
            "name": "bicycle",
            "prompts": [
                "a photo of a bicycle",
            ],
        },
        "composition": {
            "name": "boy_and_bicycle",
            "prompts": [
                "a photo of a boy and a bicycle",
            ],
        },
    },
    "sunset_ocean": {
        "concept1": {
            "name": "sunset",
            "prompts": [
                "a photo of a sunset",
            ],
        },
        "concept2": {
            "name": "ocean",
            "prompts": [
                "a photo of an ocean",
            ],
        },
        "composition": {
            "name": "sunset_and_ocean",
            "prompts": [
                "a photo of a sunset over the ocean",
            ],
        },
    },
    "desk_coffee": {
        "concept1": {
            "name": "desk",
            "prompts": [
                "a photo of a desk",
            ],
        },
        "concept2": {
            "name": "coffee",
            "prompts": [
                "a photo of a cup of coffee",
            ],
        },
        "composition": {
            "name": "desk_and_coffee",
            "prompts": [
                "a photo of a cup of coffee on a desk",
            ],
        },
    },
}
