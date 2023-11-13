import asyncio


async def str_callback_to_stream_generator(task, callback):
    queue = asyncio.Queue()

    async def new_callback(chunk: str):
        await callback(chunk)
        await queue.put(chunk)

    async def new_task():
        await task(new_callback)
        await queue.put(None)

    response_task = asyncio.create_task(new_task())

    done_response = False
    done_chunks = False

    while True:
        chunk_task = asyncio.create_task(queue.get())

        done, _pending = await asyncio.wait(
            [response_task, chunk_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        if response_task in done:
            response_task.result()
            done_response = True

        chunk = chunk_task.result() if chunk_task in done else await chunk_task

        if chunk is None:
            done_chunks = True
        else:
            yield chunk

        if done_response and done_chunks:
            break
