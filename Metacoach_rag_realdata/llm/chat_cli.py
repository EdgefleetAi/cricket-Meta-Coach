# chat_cli.py

from llm.coach_engine import coach_with_qwen

print("\nMetaCoach CLI Ready.")
print("Type 'exit' to quit.\n")

while True:
    query = input("Ask MetaCoach: ")

    if query.lower() == "exit":
        break

    answer = coach_with_qwen(query)

    print("\n========== META COACH ==========\n")
    print(answer)
    print("\n================================\n")
