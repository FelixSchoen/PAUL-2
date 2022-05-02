from src.preprocessing.data_pipeline import Tokenizer, Detokenizer
from src.config.settings import VALID_TIME_SIGNATURES


def test_tokenization():
    tokenizer = Tokenizer()
    messages = []
    generated_tokens = []

    generated_tokens.extend([0, 1, 2])

    # Append all types of messages
    for token in range(1, 25):
        messages.append({"message_type": "wait", "time": token})

    for token in range(21, 109):
        messages.append({"message_type": "note_on", "note": token})

    for token in range(21, 109):
        messages.append({"message_type": "note_off", "note": token})

    for signature in VALID_TIME_SIGNATURES:
        messages.append({"message_type": "time_signature", "numerator": signature[0], "denominator": signature[1]})

    # Tokenize each message
    for message in messages:
        tokens = tokenizer.tokenize(message)
        tokens.extend(tokenizer.flush_wait_buffer())

        assert len([i for i in generated_tokens if i in tokens]) == 0

        generated_tokens.extend(tokens)
        assert len([i for i in generated_tokens if i in tokens]) > 0

    # Assert each message included
    for token in range(0, 218):
        assert token in generated_tokens

    # Assert exactly these messages included
    assert len(generated_tokens) == 218


def test_detokenization():
    detokenizer = Detokenizer()
    tokens = range(0, 218)
    generated_messages = []

    for token in tokens:
        messages = detokenizer.detokenize(token)
        messages.extend(detokenizer.flush_wait_buffer())

        assert len([i for i in generated_messages if i in messages]) == 0

        generated_messages.extend(messages)
        assert len([i for i in generated_messages if i in messages]) > 0 or len(messages) == 0

    # Assert length
    assert len(generated_messages) == 215
