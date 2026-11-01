"""
Ensures data/profile.json is populated with the user's CV (CV_the_general_one.pdf)
as required for M4 evaluation.
"""

from pathlib import Path
from jsb.agents.profile_analyzer import analyze_profile, load_saved_profile

cv_path = Path("data/samples/cvs/CV_the_general_one.pdf")
print(f"Analyzing user CV from {cv_path}...")
profile = analyze_profile(cv_path, save_to_disk=True)
print(f"Profile saved to data/profile.json successfully!")
print(f"Name: {profile.name}")
print(f"Headline: {profile.headline}")
print(f"Years of Experience: {profile.years_experience}")
print(f"Skills ({len(profile.skills)}): {profile.skills[:10]}...")
