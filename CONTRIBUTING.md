# Contributing

Thanks for taking a look. BondSight is a small project, so the simplest way to help is to open an
issue.

- **Something doesn't work** (install, a crash, a wrong result): use the *Report a problem* form
  and include what you ran and what you saw.
- **You'd like to try it on your parts**: use the *Pilot on my line* form. Please don't attach
  photos you aren't allowed to share; a short description is enough to start.
- **Code changes**: open an issue first for anything bigger than a small fix, so we can agree on
  the approach. Run the tests before sending a pull request:

  ```bash
  inspector/.venv/Scripts/python -m pytest inspector/tests -q
  synth/.venv/Scripts/python -m pytest synth/tests -q
  ```

- **Data**: training data must be under a licence that allows redistribution with credit
  (CC BY, CC0, BY-SA). Non-commercial (NC) and no-derivatives (ND) data can't be used. Never commit
  third-party photos; add the source to [DATA_SOURCES.md](DATA_SOURCES.md) and the credit to
  [CREDITS.md](CREDITS.md) instead.

By contributing you agree that your code is released under the MIT licence and your text and
pictures under CC BY 4.0, like the rest of the project.
