from src import looks_like_command

KW_NOUNS = ["кредит", "карта", "комиссия", "погашение", "лимит", "перевод", "ипотека"]


if __name__ == '__main__':
    result = looks_like_command(text='Выдайте мне ипотеку.',
                                kw_nouns=KW_NOUNS)
    print(result)

