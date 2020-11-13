define([
    './properties',
    'text!./hello-world.ng.html',
    'css!./hello-world.ng.css'
  ],
  function (props, template, style) {
    'use strict';

    return {
      definition: props,
      template: template,
      style: style,
      controller: ['$scope', function ($scope) {
        console.log('layout', $scope.layout);
      }]
    };
  });