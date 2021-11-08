import re

# You are given a number input, and need to check if it is a valid phone number.
# A valid phone number has exactly 8 digits and starts with 1, 8 or 9.
# Output "Valid" if the number is valid and "Invalid", if it is not.


def phone_number(number):
    if re.match(r'^(1|8|9)[0-9]{7}$', number):
        print("Valid")
    else:
        print("Invalid")


phone_number("13230517")
