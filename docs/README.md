# Documentation

sphinx를 이용하여 제작한 코드 설명 문서. 함수의 인자, 인자 타입, 설명 등 Google Style Python Docstring을 Rendering하여 출력해 줌.

또한 Docstring에는 필요한 경우 LATEX를 사용한 수식이 삽입 되어 있어, text로 표현하기 어려운 수식을 시각적으로 확인해 볼수 있음.

## Requirements

* Sphinx

    `pip install sphinx`

    `pip install sphinxcontrib-napoleon`

## Build

문서를 제작하기 위해 `/docs/1_gen-docs.sh`를 실행하면 문서 빌드가 진행되고 `/docs/build` 디렉토리가 생성된다.

그 후 `/docs/build/html/index.html`을 실행해서 문서를 볼수 있다.
