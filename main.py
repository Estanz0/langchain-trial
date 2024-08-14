import json
import os
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from rich import print

load_dotenv()

COMPANY_NAME = os.getenv("COMPANY_NAME")


# Define your desired data structure.
class LevelEnum(str, Enum):
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"


class Skill(BaseModel):
    name: str = Field(description="skill's name")
    level: LevelEnum = Field(description="skill's level")


class UserInputHistory(BaseModel):
    query: str = Field(description="user's query")
    field: str = Field(description="JSON field that was updated. For example, 'skills -> SQL'")
    old_value: str | LevelEnum = Field(description="old value. For example, 'beginner'")
    new_value: str | LevelEnum = Field(description="new value. For example, 'intermediate'")


class User(BaseModel):
    name: str = Field(description="user's name")
    age: int = Field(description="user's age")
    email: str = Field(description="user's email")
    skills: list[Skill] = Field(description="user's skills")


class UserCreate(User): ...


class UserUpdate(User): ...


class UserQuery(BaseModel):
    rsponse: str = Field(description="response")


# Model
model = ChatOpenAI(temperature=0)

# Prompts
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query", "format_instructions"],
    partial_variables={
        "General Context": f"This is a user profile of an employee at {COMPANY_NAME}."
        "It will be used to match the user with the right job."
    },
)


# Step 1: Create initial user profile
def create_user_profile(user_info: str) -> UserCreate:
    user_create_query = f"Create a user with the following information:\n{user_info}\n"

    parser = JsonOutputParser(pydantic_object=UserCreate)
    chain = prompt | model | parser

    return chain.invoke({"query": user_create_query, "format_instructions": parser.get_format_instructions()})


# Step 2: Update user profile
def update_user_profile(user_profile: User, user_input: str) -> User:
    user_update_query = f"Update the user's profile with new or updated skills:\n{user_profile}"
    user_update_query += f"User Input: {user_input}"

    parser = JsonOutputParser(pydantic_object=UserUpdate)
    chain = prompt | model | parser

    return chain.invoke({"query": user_update_query, "format_instructions": parser.get_format_instructions()})


def query_user_profile(user_profile: User, input_query: str) -> UserQuery:
    user_query = f"Get and explain the relevant information from the user profile:\n{user_profile}"
    user_query += f"Relating to: {input_query}"

    parser = JsonOutputParser(pydantic_object=UserQuery)
    chain = prompt | model | parser

    return chain.invoke({"query": user_query, "format_instructions": parser.get_format_instructions()})


with Path("user_info.txt").open() as file:
    user_info = file.read()

user_profile = create_user_profile(user_info)
print(user_profile)

user_input = "I just completed a full stack website with 1000 users using react and tailwind css."
user_profile_new = update_user_profile(user_profile=user_profile, user_input=user_input)
print(user_profile_new)

input_query = "Is this person a good fit for a solution architect role?"
user_query = query_user_profile(user_profile=user_profile_new, input_query=input_query)
print(user_query)
