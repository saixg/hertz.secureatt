from database import create_db_and_tables, seed_data

# Force create database and seed data
create_db_and_tables()
seed_data()
print("Database created and seeded!")