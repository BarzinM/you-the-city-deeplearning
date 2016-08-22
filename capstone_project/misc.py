from time import time
import inspect


def breakLine(text, wrap=80):
    if len(text) > wrap:
        char = wrap
        while char > 0 and text[char] != ' ':
            char -= 1
        if char:
            text = [text[:char]] + breakLine(text[char + 1:], wrap)
        else:
            text = [text[:wrap - 1] + '-'] + breakLine(text[wrap - 1:], wrap)
        return text
    else:
        return [cleanLine(text)]


def cleanLine(text):
    if text[-1] == ' ':
        text = text[:-1]
    if text[0] == ' ':
        text = text[1:]
    return text


def boxPrint(text, wrap=0):
    line_style = '-'
    paragraph = text.split('\n')
    if wrap > 0:
        index = 0
        while index < len(paragraph):
            paragraph[index] = cleanLine(paragraph[index])
            if len(paragraph[index]) > wrap:
                paragraph = paragraph[
                    :index] + breakLine(paragraph[index], wrap) + paragraph[index + 1:]
            index += 1

    length = (max([len(line) for line in paragraph]))
    print('+' + line_style * length + '+')
    for line in paragraph:
        print('|' + line + ' ' * (length - len(line)) + '|')
    print('+' + line_style * length + '+')


class Profiler(object):
    def __init__(self, string='Intialization'):
        self.lines = [((string, self.__getFrame()[0]))]
        self.times = [time()]

    def __getFrame(self):
        line_number, function_name = inspect.getouterframes(
            inspect.currentframe())[2][2:4]
        return line_number, function_name

    def profile(self, string=''):
        self.times.append(time())
        if string is '':
            string = self.__getFrame()[1] + '()'
        self.lines.append((string, self.__getFrame()[0]))
        self.times.append(time())  # to reduce overhead of profiler

    def end(self, string='End'):
        self.times.append(time())
        self.lines.append((string, self.__getFrame()[0]))

    def report(self):
        t = time()
        times = self.times
        lines = self.lines
        # print([line[0] for line in lines[:,0]])
        width = min(max([len(line[0]) for line in lines]), 20)
        if len(times) % 2 == 1:
            lines.append(('End', self.__getFrame()[0]))
        times.append(t)
        text = ''
        for i in range(len(lines) - 1):
            name = lines[i][0]
            name = max(width - len(name), 0) * " " + lines[i][0][:width]
            text += "%s @ %i to %i: %.3f s\n" % (name, lines[i][1], lines[
                i + 1][1], times[2 * i + 1] - times[2 * i])
        boxPrint(text[:-1])


def do():
    m = Profiler()
    import numpy
    m.profile('adding a profile from this line')
    numpy.random.rand(200000)
    print('doing something ..')
    print(m.times)
    m.profile()
    print('doing some other thing')
    numpy.arange(20000000)
    print('blah blah')
    m.end()  # can be ommitted
    m.report()

if __name__ == "__main__":
    do()
