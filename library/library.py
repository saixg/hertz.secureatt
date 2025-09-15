import os

Library = []

def add_book(title, author, genre, year, status="unread"):
    book = {
        "title": title.strip(),
        "author": author.strip(),
        "genre": genre.strip(),
        "year": int(year),
        "status": status.lower()
    }
    Library.append(book)
    print(f"Book '{title}' added.")

def list_books(sorted_by_year=False):
    if not Library:
        print("Library is empty.")
        return

    books = sorted(Library, key=lambda x: x['year']) if sorted_by_year else Library

    for idx, book in enumerate(books, start=1):
        print(f"{idx}. {book['title']} by {book['author']} [{book['genre']}], {book['year']}")

def search_by_author(author_name):
    return [book for book in Library if author_name.lower() in book['author'].lower()]

def delete_book(title):
    global Library
    original_len = len(Library)
    Library = list(filter(lambda book: book['title'].lower() != title.lower(), Library))
    if len(Library) < original_len:
        print(f"Book titled '{title}' has been removed.")
    else:
        print(f"No book titled '{title}' found.")

def show_reading_summary():
    if not Library:
        print("No books in library.")
        return

    read = len([book for book in Library if book['status'] == 'read'])
    unread = len(Library) - read

    print(f"Read: {read} books ({(read / len(Library)) * 100:.2f}%)")
    print(f"Unread: {unread} books ({(unread / len(Library)) * 100:.2f}%)")

def uppercase_titles():
    titles = [book['title'].upper() for book in Library]
    print("Book Titles (UPPERCASE):")
    for title in titles:
        print("-", title)

def save_to_file(filename):
    try:
        with open(filename, 'w') as f:
            for book in Library:
                line = f"{book['title']},{book['author']},{book['genre']},{book['year']},{book['status']}\n"
                f.write(line)
        print(f"Saved {len(Library)} books to {filename}")
    except Exception as e:
        print("Error saving file:", e)

def load_from_file(filename):
    if not os.path.exists(filename):
        print("File not found. Starting with an empty library.")
        return
    try:
        with open(filename, 'r') as f:
            for line in f:
                title, author, genre, year, status = line.strip().split(',')
                add_book(title, author, genre, int(year), status)
    except Exception as e:
        print("Error reading file:", e)

def menu():
    while True:
        print("\n--- Personal Library Menu ----")
        print("1. Add Book")
        print("2. List All Books")
        print("3. List Books Sorted by Year")
        print("4. Search by Author")
        print("5. Delete Book by Title")
        print("6. Show Reading Summary")
        print("7. Show UPPERCASE Book Titles (map)")
        print("8. Save Library")
        print("9. Load Library")
        print("10. Exit")

        choice = input("Choose an option: ")

        if choice == "1":
            title = input("Title: ")
            author = input("Author: ")
            genre = input("Genre: ")
            year = input("Year: ")
            status = input("Status (read/unread): ")
            add_book(title, author, genre, year, status)
        elif choice == "2":
            list_books()
        elif choice == "3":
            list_books(sorted_by_year=True)
        elif choice == "4":
            author = input("Enter author name: ")
            results = search_by_author(author)
            print(f"\nFound {len(results)} books by {author}:")
            for book in results:
                print(f"- {book['title']} ({book['year']})")
        elif choice == "5":
            title = input("Enter title to delete: ")
            delete_book(title)
        elif choice == "6":
            show_reading_summary()
        elif choice == "7":
            uppercase_titles()
        elif choice == "8":
            save_to_file("books_data.txt")
        elif choice == "9":
            load_from_file("books_data.txt")
        elif choice == "10":
            print("Goodbye!")
            break
        else:
            print("Invalid option. Try again.")

if __name__ == "__main__":
    menu()
