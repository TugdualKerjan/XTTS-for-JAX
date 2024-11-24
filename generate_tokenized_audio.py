from datasets import load_dataset, DatasetDict, Dataset
import tqdm
from mytokenizer import Tokenizer

dataset = load_dataset("blabble-io/libritts_r", "clean", streaming=True)


tokenizer = Tokenizer(checkpoint_dir="checkpoints", sample_rate=22050)


def encode(sample):
    # print(sample)
    text, audio = tokenizer.encode(
        sample["text_normalized"],
        sample["audio"]["array"],
        sample["audio"]["sampling_rate"],
    )
    print(text)

    return {"text": text, "audio": audio}


def remove_too_long(sample):
    return len(sample["text"]) + len(sample["audio"]) <= 200


def pad(sample):
    return {
        "audio": tokenizer.pad(
            sample["audio"], len(sample["text"]) + len(sample["audio"])
        )
    }


dataset = (
    dataset.map(
        encode,
        remove_columns=[
            "text_original",
            "speaker_id",
            "chapter_id",
            "id",
            "path",
            "text_normalized",
        ],
    )
    .filter(remove_too_long)
    .map(pad)
)

samples = []
for i, sample in enumerate(dataset["train.clean.360"]):
    if i >= 10000:
        break
    samples.append(sample)

# Create a Dataset from the collected samples
dataset = Dataset.from_dict(
    {key: [sample[key] for sample in samples] for key in samples[0].keys()}
)

# Convert to DatasetDict
dataset_dict = DatasetDict({"train": dataset})

# Save the dataset to disk
dataset_dict.save_to_disk("preprocessed_dataset")

print("Saved the preprocessed dataset to preprocessed_dataset")
