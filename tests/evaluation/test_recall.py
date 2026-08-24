from typing import List, Set


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:

    retrieved = set(retrieved_ids[:k])
    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    hit = retrieved & relevant

    return len(hit) / len(relevant)


retrieved = ["chunk_10", "chunk_2", "chunk_8", "chunk_3", "chunk_20"]

relevant = ["chunk_2", "chunk_3"]

print(recall_at_k(retrieved, relevant, 5))
