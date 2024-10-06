# version: major.minor.patch+commit-hash

MAJOR = 0
MINOR = 0
PATCH = 5

# if commit hash is defined in env:HACCA_COMMIT_HASH
# then add it to the version
__version__ = f"{MAJOR}.{MINOR}.{PATCH}"
