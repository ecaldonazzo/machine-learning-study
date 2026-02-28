#!/bin/sh
chown -R appuser:appuser /opt/project/PythonProjects/report
chown -R appuser:appuser /opt/project/PythonProjects/models
exec gosu appuser "$@"