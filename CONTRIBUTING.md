
### Contributing

Steps to do to contribute:
1. Read [documentations](atradebot.readthedocs.io)
2. Create a fork.
3. Write code.
   - Add new data input source in [news_utils.py](https://github.com/Andrechang/Atradebot/blob/main/src/atradebot/news_utils.py)
   - Add new task dataset in [fin_data.py](https://github.com/Andrechang/Atradebot/blob/main/src/atradebot/fin_data.py)
   - Add new trade strategy in [strategies.py](https://github.com/Andrechang/Atradebot/blob/main/src/atradebot/strategies.py)
   - Add new generic helper functions in [utils.py](https://github.com/Andrechang/Atradebot/blob/main/src/atradebot/utils.py)
   - Add new tests in [tests](https://github.com/Andrechang/Atradebot/tree/main/tests)
   - Add documentation in [docs](https://github.com/Andrechang/Atradebot/tree/main/docs)
        
4. Write comments.
For each function add the comments according to [sphinx](https://www.sphinx-doc.org/en/master/) documentation format.
Check how documentation of additions will look like by running:
```
cd docs/
sphinx-build -b html
```

5. Write commit and pull request description.
Commit and pull request description should contain what the addition does and how to test it.

6. Make sure it doesnt break other things.
Run test using:
```
pytest ./tests/test_main.py
```

7. Make pull request and wait for review.

8. Congrats and thank you
